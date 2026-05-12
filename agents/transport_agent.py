"""
交通专家智能体 TransportAgent（纯规则版，无 LLM）

数据源：
  - 火车：mcp_clients.train_client（12306）
  - 航班：mcp_clients.tuniu_client（途牛 CLI）
  - 天气：mcp_clients.amap_client.get_city_weather（高德 maps_weather）

设计：
  - 三路（含火车票价共四次调用）并发拉取，任何一路失败仅令对应 options 为空
  - 飞机 / 火车各按价升序取 top 3 → 合并为 options[]
  - is_recommended：同时打在"飞机最低价"和"火车最低价"两条上（若该类存在）
  - recommendation.arrival_hub：取全局最低价那条的 arrival_hub（accommodation_agent 唯一消费方）

输出契约（保持与旧版兼容，前端 TransportTable 仅消费 options 列各字段 + is_recommended）：
  {
      "transport_plan": {
          "query_info":      {origin, destination, date, data_source},
          "options":         [TravelOption-like dict, ...],
          "recommendation":  {best_choice, arrival_hub, reason},
          "weather_summary": "晴 18-27℃"  (可空字符串)
      }
  }
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from mcp_clients.train_client import train_client
from mcp_clients.tuniu_client import (
    flight_search,
    iter_flights,
    normalize_flight,
    unwrap_mcp_content,
    TuniuCallError,
)
from mcp_clients.amap_client import (
    amap_mcp_session,
    get_city_weather,
    summarize_weather,
)
from utils.tuniu_budget import TuniuBudgetExceeded
from utils.date_resolver import normalize_date

logger = logging.getLogger(__name__)

# 火车席别策略：严格只用二等座。
# 12306 query_ticket_price 的 prices 字段以中文席别名为 key（如 "二等座": "553.0"）。
# 没有二等座的车次（如普速 K/T/Z 系列、夜间动卧）将 price=None，不参与推荐。
_TRAIN_SEAT_KEY = "二等座"
_TOP_N_PER_TYPE = 3   # 飞机 / 火车各取 top N 条


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _parse_price_to_int(raw: Any) -> Optional[int]:
    """各种价格字符串/数字 → int 元；解析失败返回 None。

    兼容 "¥553" / "553元" / "553.0" / 553 / "553.50" 等格式。
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        m = re.search(r"\d+(?:\.\d+)?", raw)
        if m:
            try:
                return int(float(m.group()))
            except ValueError:
                return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 段 2：数据源处理 + 选项构造（纯函数，无 IO）
# ─────────────────────────────────────────────────────────────────────────────

def _build_flight_option(raw: Dict[str, Any]) -> Dict[str, Any]:
    """将 normalize_flight() 输出转为 TravelOption-shape dict。

    输入字段（来自 mcp_clients.tuniu_client.normalize_flight）：
      flight_no / airline / dep_time / arr_time / dep_airport / arr_airport
      / duration / price(int|None) / cabin_class
    缺失字段保持 None；price_range 在有价时拼成 "¥553"，否则 None。
    """
    price = _parse_price_to_int(raw.get("price"))
    price_range = f"¥{price}" if price is not None else None

    return {
        "transport_type": "飞机",
        "transport_no":   raw.get("flight_no"),
        "departure_time": raw.get("dep_time"),
        "arrival_time":   raw.get("arr_time"),
        "duration":       raw.get("duration"),
        "departure_hub":  raw.get("dep_airport"),
        "arrival_hub":    raw.get("arr_airport"),
        "price_range":    price_range,
        "flight_company": raw.get("airline"),
        # tuniu basePrice 即 经济舱 低价；cabin_class 缺失时统一兜底
        "cabin_class":    raw.get("cabin_class") or "经济舱",
        "is_recommended": False,
        "data_source":    "realtime",
        # 非 TravelOption 字段，仅用于 _pick_recommended 排序与 recommendation 组装
        "_price_int":     price,
    }


def _coerce_json(payload: Any) -> Any:
    """把 12306 client 返回兼容成 Python 对象：JSON 字符串则 loads，否则原样。

    解析失败一律返回 None；调用方据此走"该路降级"路径。
    """
    if payload is None:
        return None
    if isinstance(payload, (dict, list)):
        return payload
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except (TypeError, ValueError):
            return None
    return None


def _merge_train_prices(trains_payload: Any, prices_payload: Any) -> List[Dict[str, Any]]:
    """合并 12306 query_tickets 与 query_ticket_price 两路返回。

    输入两侧均接受：
      - JSON 字符串（train_client 默认返回形态）
      - 已解析的 dict（success/trains 或 success/data 信封）
      - None / 异常对象（asyncio.gather return_exceptions=True 会留下 Exception 实例）

    返回：带 prices/train_code 字段的 trains list；任一侧不可用即返回 []。
    """
    trains_obj = _coerce_json(trains_payload)
    prices_obj = _coerce_json(prices_payload)

    if not isinstance(trains_obj, dict) or trains_obj.get("success") is not True:
        return []
    trains: List[Dict[str, Any]] = [
        t for t in (trains_obj.get("trains") or []) if isinstance(t, dict)
    ]
    if not trains:
        return []

    # prices 侧缺失不致命，trains 仍可返回（价格字段为 None，_build_train_option 会兜底跳过）
    # 注意：两路 API 的 train_no 含义不同：
    #   - query_tickets:      train_no = 用户可见车次号 "G547"
    #   - query_ticket_price: train_no = 12306 内部 ID，train_code = "G547"
    # 故合并键统一用"用户可见 G547"：trains.train_no ≡ prices.train_code
    price_map: Dict[str, Dict[str, Any]] = {}
    if isinstance(prices_obj, dict) and prices_obj.get("success") is True:
        for row in prices_obj.get("data") or []:
            if not isinstance(row, dict):
                continue
            key = row.get("train_code")
            if not key:
                continue
            price_map[key] = row

    merged: List[Dict[str, Any]] = []
    for t in trains:
        key = t.get("train_no")
        price_row = price_map.get(key) if key else None
        item = dict(t)  # 浅拷贝，避免改原始 dict
        if price_row:
            item["prices"] = price_row.get("prices") or {}
        else:
            item.setdefault("prices", {})
        merged.append(item)

    return merged


def _build_train_option(train: Dict[str, Any]) -> Dict[str, Any]:
    """将 _merge_train_prices 单条输出转为 TravelOption-shape dict。

    数据来源字段（query_tickets 主体 + prices 合并）：
      train_no / from_station / to_station / start_time / arrive_time / duration
      / prices: Dict[中文席别名, "23.0"|""]
    取价策略：严格只用 _TRAIN_SEAT_KEY（二等座）。没有二等座或解析失败的车次
              price_range / cabin_class / _price_int 均为 None，
              _pick_recommended 与全局最低价排序会自动跳过。
    """
    prices: Dict[str, Any] = train.get("prices") or {}

    price_range: Optional[str] = None
    price_int: Optional[int] = None
    cabin_class: Optional[str] = None
    p = _parse_price_to_int(prices.get(_TRAIN_SEAT_KEY))
    if p is not None and p > 0:
        price_range = f"¥{p}"
        price_int = p
        cabin_class = _TRAIN_SEAT_KEY

    return {
        "transport_type": "火车",
        "transport_no":   train.get("train_no"),
        "departure_time": train.get("start_time"),
        "arrival_time":   train.get("arrive_time"),
        "duration":       train.get("duration"),
        "departure_hub":  train.get("from_station"),
        "arrival_hub":    train.get("to_station"),
        "price_range":    price_range,
        "cabin_class":    cabin_class,
        "is_recommended": False,
        "data_source":    "realtime",
        "_price_int":     price_int,
    }


def _pick_recommended(options: List[Dict[str, Any]]) -> None:
    """飞机 / 火车两类各自最低价那条 is_recommended=True（原地修改）。

    - 空列表 / 全类型 _price_int 均缺失：直接返回，不修改
    - 同价并列：取 options 中首次出现的那条（稳定）
    """
    if not options:
        return
    by_type: Dict[str, Dict[str, Any]] = {}   # type -> 当前最低价那条
    for opt in options:
        price = opt.get("_price_int")
        if price is None:
            continue
        t = opt.get("transport_type")
        best = by_type.get(t)
        if best is None or price < best["_price_int"]:
            by_type[t] = opt
    for opt in by_type.values():
        opt["is_recommended"] = True


# ─────────────────────────────────────────────────────────────────────────────
# 段 3：TransportAgent 主编排
# ─────────────────────────────────────────────────────────────────────────────

class TransportAgent:
    """纯规则版交通查询：四路并发 → 选项组装 → 推荐。无 LLM。

    兼容旧契约：`run(input_data: dict)`，从 input_data["context"]["key_entities"]
    取 origin / destination / date；返回 {"transport_plan": {...}} 或 {"error": ...}。
    """

    def __init__(self, name: str = "TransportAgent", model: Any = None, **_kwargs: Any):
        # model 形参仅为与旧构造签名兼容（accommodation_agent / cli.py 仍可能传），
        # 本 agent 不调用 LLM，故不持有引用。
        self.name = name

    # ── 数据拉取协程（供 asyncio.gather 调度） ────────────────────────

    @staticmethod
    async def _fetch_flights(origin: str, destination: str, date: str) -> List[Dict[str, Any]]:
        """tuniu flight_search → MCP 信封 → 列表 → normalize。

        任意一步异常向上抛，由 run() 的 return_exceptions 捕获。
        正常返回 [normalize_flight(...) ...]，空结果返回 []。
        """
        raw = await flight_search(
            departure_city=origin,
            arrival_city=destination,
            departure_date=date,
        )
        unwrapped = unwrap_mcp_content(raw)
        return [normalize_flight(f) for f in iter_flights(unwrapped)]

    @staticmethod
    async def _fetch_weather(destination: str) -> str:
        """高德 maps_weather → 一行简报。任意失败向上抛。"""
        async with amap_mcp_session() as session:
            data = await get_city_weather(session, destination)
        return summarize_weather(data)

    # ── 主入口 ──────────────────────────────────────────────────────────

    async def run(self, input_data: dict) -> dict:
        context = input_data.get("context", {}) or {}
        key_entities = context.get("key_entities", {}) or {}

        origin      = (key_entities.get("origin") or "").strip()
        destination = (key_entities.get("destination") or "").strip()
        date_raw    = (key_entities.get("date") or "").strip()

        missing: List[str] = []
        if not origin:
            missing.append("出发地")
        if not destination:
            missing.append("目的地")
        if missing:
            return {"error": f"缺少{'和'.join(missing)}，请补充后再查询。"}

        # 日期：缺失补明天；中文/斜杠 → YYYY-MM-DD
        if not date_raw:
            date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info(f"[TransportAgent] date 缺失，默认明天 {date}")
        else:
            date = normalize_date(date_raw) or date_raw

        # ── 四路并发；return_exceptions=True 让某一路失败不拖垮其他三路 ──
        train_raw, price_raw, flights_raw, weather_raw = await asyncio.gather(
            train_client.query_tickets(date, origin, destination),
            train_client.query_ticket_price(date, origin, destination),
            self._fetch_flights(origin, destination, date),
            self._fetch_weather(destination),
            return_exceptions=True,
        )

        # ── 火车 options 组装 ───────────────────────────────────────────
        # train_client 的两个查询不直接抛异常，失败时返回字符串错误信息；
        # 但 gather(return_exceptions=True) 防御性兜底未预期异常。
        train_options: List[Dict[str, Any]] = []
        if isinstance(train_raw, Exception) or isinstance(price_raw, Exception):
            logger.warning(
                f"[TransportAgent] 火车查询失败 train_err={train_raw if isinstance(train_raw, Exception) else None} "
                f"price_err={price_raw if isinstance(price_raw, Exception) else None}"
            )
        else:
            merged = _merge_train_prices(train_raw, price_raw)
            train_options = [_build_train_option(t) for t in merged]
            train_options.sort(key=lambda o: o.get("_price_int") if o.get("_price_int") is not None else float("inf"))
            train_options = train_options[:_TOP_N_PER_TYPE]

        # ── 飞机 options 组装 ───────────────────────────────────────────
        flight_options: List[Dict[str, Any]] = []
        if isinstance(flights_raw, Exception):
            if isinstance(flights_raw, (TuniuBudgetExceeded, TuniuCallError)):
                logger.warning(f"[TransportAgent] tuniu 航班查询失败: {flights_raw}")
            else:
                logger.warning(f"[TransportAgent] 航班查询未预期异常: {flights_raw!r}")
        else:
            flight_options = [_build_flight_option(f) for f in flights_raw]
            flight_options.sort(key=lambda o: o.get("_price_int") if o.get("_price_int") is not None else float("inf"))
            flight_options = flight_options[:_TOP_N_PER_TYPE]

        # ── 天气 ────────────────────────────────────────────────────────
        if isinstance(weather_raw, Exception):
            logger.info(f"[TransportAgent] 天气查询失败（可忽略）: {weather_raw}")
            weather_summary = ""
        else:
            weather_summary = weather_raw or ""

        # ── 合并 + 推荐 ─────────────────────────────────────────────────
        # 顺序：飞机在前、火车在后；前端按列表顺序渲染，飞机优先曝光（多数城际场景更快）
        options: List[Dict[str, Any]] = flight_options + train_options
        _pick_recommended(options)

        # 全局最低价 → recommendation（best_choice / arrival_hub）
        best_choice: Optional[str] = None
        arrival_hub: Optional[str] = None
        reason: str = ""
        priced = [o for o in options if o.get("_price_int") is not None]
        if priced:
            best = min(priced, key=lambda o: o["_price_int"])
            no = best.get("transport_no") or "-"
            price = best["_price_int"]
            best_choice = f"{no} ¥{price}"
            arrival_hub = best.get("arrival_hub")
            reason = "全局最低价"

        # 出参剥离内部字段，保持给前端/校验器的 dict 干净
        for opt in options:
            opt.pop("_price_int", None)

        return {
            "transport_plan": {
                "query_info": {
                    "origin":      origin,
                    "destination": destination,
                    "date":        date,
                    "data_source": "12306 + tuniu",
                },
                "options":         options,
                "recommendation": {
                    "best_choice": best_choice,
                    "arrival_hub": arrival_hub,
                    "reason":      reason,
                },
                "weather_summary": weather_summary,
            }
        }
