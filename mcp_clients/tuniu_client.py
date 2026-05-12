"""
途牛 MCP CLI 客户端

通过 subprocess 调用 `tuniu call <server> <tool> --args <json> --output json`，
统一处理鉴权（TUNIU_API_KEY 由父进程 env 继承）、超时、错误分类与结果缓存。

设计：
- 失败一律抛 TuniuCallError(type, code, message, details)，不返回 error dict
- 限流/配额由 utils.tuniu_budget 单例兜底；超额抛 TuniuBudgetExceeded（不转 TuniuCallError）
- 每次调用方传入 ttl_sec，绕过 with_budget 装饰器以支持差异化缓存
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from typing import Any, Optional

from config import TUNIU_MCP_CONFIG
from utils.tuniu_budget import tuniu_budget

logger = logging.getLogger(__name__)

# 模块级缓存：避免每次调用都 shutil.which
_RESOLVED_CMD: Optional[str] = None


class TuniuCallError(Exception):
    """途牛 CLI 调用失败的结构化异常。

    type 取值：
      - cli_failure  : stdout 解析出 success:false，code/message 来自 error 块
      - process_error: 进程非零退出且 stdout 不是合法错误 JSON（含 CLI 未找到）
      - timeout      : asyncio.wait_for 超时
      - parse_error  : stdout 非合法 JSON 或结构异常
    """

    def __init__(
        self,
        type: str,
        message: str,
        code: str | int | None = None,
        details: dict | None = None,
    ):
        self.type = type
        self.code = code
        self.message = message
        self.details = details or {}
        suffix = f" (code={code})" if code is not None else ""
        super().__init__(f"[{type}] {message}{suffix}")


def _resolve_cmd() -> str:
    """惰性解析 tuniu 可执行路径；找不到抛 process_error"""
    global _RESOLVED_CMD
    if _RESOLVED_CMD is not None:
        return _RESOLVED_CMD
    raw = TUNIU_MCP_CONFIG["command"]
    resolved = shutil.which(raw) or raw
    # 如果 raw 是绝对路径但不存在，shutil.which 返回 None，这里 fallback 原值由 exec 阶段抛 FileNotFoundError
    _RESOLVED_CMD = resolved
    return resolved


async def _run_tuniu_cli(
    server: str,
    tool: str,
    args: dict | None,
    timeout: float,
) -> Any:
    """执行 `tuniu call <server> <tool> --args <json> --output json` 并返回 data 字段。

    失败一律抛 TuniuCallError；调用方按 type 区分处理。

    实现注记：使用同步 subprocess.run + asyncio.to_thread 而非 asyncio.create_subprocess_exec。
    原因：uvicorn --reload 在 Windows 下会强制使用 SelectorEventLoop，
    而 SelectorEventLoop 不支持子进程，会抛 NotImplementedError。
    走线程池里的同步 subprocess 与事件循环类型解耦，开发/生产都能跑。
    """
    cmd = _resolve_cmd()
    payload = json.dumps(args or {}, ensure_ascii=False)
    argv = [cmd, "call", server, tool, "--args", payload, "--output", "json"]

    def _blocking_run() -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            argv,
            capture_output=True,
            timeout=timeout,
            check=False,
        )

    # 1. 启动子进程并等待结果
    #    FileNotFoundError → process_error
    #    TimeoutExpired   → timeout
    try:
        completed = await asyncio.to_thread(_blocking_run)
    except FileNotFoundError:
        raise TuniuCallError(
            "process_error",
            f"tuniu CLI 未找到: {cmd}",
            details={"argv": argv},
        )
    except subprocess.TimeoutExpired:
        raise TuniuCallError(
            "timeout",
            f"tuniu {server}.{tool} 超过 {timeout}s",
        )

    stdout = completed.stdout.decode("utf-8", errors="replace").strip()
    stderr = completed.stderr.decode("utf-8", errors="replace").strip()

    # 3. 解析 stdout JSON（非法 → parse_error）
    try:
        data = json.loads(stdout) if stdout else None
    except json.JSONDecodeError as e:
        raise TuniuCallError(
            "parse_error",
            f"CLI 输出非合法 JSON: {e}",
            details={
                "stdout": stdout[:500],
                "stderr": stderr[:500],
                "returncode": completed.returncode,
            },
        )

    # 4. 业务失败（success:false） → cli_failure，优先于 returncode 判定
    if isinstance(data, dict) and data.get("success") is False:
        err = data.get("error") or {}
        raise TuniuCallError(
            "cli_failure",
            err.get("message", "tuniu CLI returned failure"),
            code=err.get("code"),
            details={
                "error_type": err.get("type"),
                "error_details": err.get("details"),
                "returncode": completed.returncode,
            },
        )

    # 4.5 业务成功但进程在退出阶段崩溃（Windows libuv 已知断言：
    #     stdout 已完整写出 success:true，仅在 cleanup 阶段触发 UV_HANDLE_CLOSING）
    #     此时数据是可信的，仅记 warning，不视为失败。
    if (
        isinstance(data, dict)
        and data.get("success") is True
        and completed.returncode != 0
    ):
        logger.warning(
            "tuniu %s.%s 业务成功但进程异常退出 returncode=%s stderr=%s",
            server, tool, completed.returncode, stderr[:200],
        )
    elif completed.returncode != 0:
        # 5. 进程异常退出且无业务结果 → process_error
        raise TuniuCallError(
            "process_error",
            f"tuniu 退出码={completed.returncode}",
            code=completed.returncode,
            details={"stderr": stderr[:500], "stdout": stdout[:500]},
        )

    # 5. 进程异常退出但 stdout 没给出业务错误 → process_error
    elif completed.returncode != 0:
        raise TuniuCallError(
            "process_error",
            f"tuniu 退出码={completed.returncode}",
            code=completed.returncode,
            details={"stderr": stderr[:500], "stdout": stdout[:500]},
        )

    # 6. 结构校验：必须是带 success 字段的 dict
    if not isinstance(data, dict) or "success" not in data:
        raise TuniuCallError(
            "parse_error",
            "CLI 输出缺少 success 字段",
            details={"stdout": stdout[:500]},
        )

    # 7. 成功：优先 result（CLI 实际包装字段）→ data（兜底）→ 剩余字段
    for key in ("result", "data"):
        if key in data:
            return data[key]
    return {k: v for k, v in data.items() if k != "success"}


async def call_tuniu(
    server: str,
    tool: str,
    args: Optional[dict] = None,
    ttl_sec: float = 0,
    timeout: Optional[float] = None,
) -> Any:
    """统一调用入口：缓存命中即返；未命中走预算闸门 + CLI；成功后按 ttl_sec 落缓存。

    Args:
        server: 途牛 MCP 域名，如 'hotel' / 'flight'
        tool:   工具名，如 'tuniu_hotel_search'
        args:   工具入参 dict；None 视同 {}
        ttl_sec: 命中后缓存时长（秒）；<=0 表示不缓存。差异化 TTL 由调用方传入
        timeout: 单次 CLI 超时（秒）；None 时使用 TUNIU_MCP_CONFIG['timeout']

    Returns:
        CLI 返回 JSON 中的 data 字段（或除 success 外的剩余字段）

    Raises:
        TuniuCallError:        CLI 调用本身失败（4 种 type，见类定义）
        TuniuBudgetExceeded:   RPM 等待超时或 RPD 已用满（由 tuniu_budget.acquire 抛出，
                              上层据此区分"限流"与"调用失败"）
    """
    key = tuniu_budget.make_cache_key(server, tool, args)

    cached = tuniu_budget.cache_get(key)
    if cached is not None:
        logger.debug("tuniu cache hit: %s", key)
        return cached

    # 闸门：限流/配额超额直接抛 TuniuBudgetExceeded，不进入子进程
    await tuniu_budget.acquire()

    effective_timeout = timeout if timeout is not None else TUNIU_MCP_CONFIG["timeout"]
    logger.info("tuniu call: %s.%s args=%s", server, tool, args)

    # CLI 失败抛 TuniuCallError，向上传播，不写缓存
    result = await _run_tuniu_cli(server, tool, args, effective_timeout)

    tuniu_budget.cache_set(key, result, ttl_sec)
    return result


# ---------------- 高层封装 ----------------

def _ttl_for(server: str, tool: str) -> float:
    """从 TUNIU_MCP_CONFIG['cache_ttl'] 查 'server:tool' 的 TTL；缺省 0（不缓存）"""
    return TUNIU_MCP_CONFIG.get("cache_ttl", {}).get(f"{server}:{tool}", 0)


async def hotel_search(
    city_name: str,
    check_in: Optional[str] = None,
    check_out: Optional[str] = None,
    keyword: Optional[str] = None,
    prices: Optional[str] = None,
    query_id: Optional[str] = None,
    page_num: Optional[int] = None,
) -> Any:
    """途牛酒店搜索 hotel.tuniu_hotel_search。

    必填：city_name。翻页传 query_id + page_num（首次搜索返回 queryId）。
    返回 CLI result/data 原始结构，字段解析见 Section C。
    """
    if not city_name:
        raise ValueError("hotel_search: city_name 不能为空")

    args: dict[str, Any] = {"cityName": city_name}
    # 按需注入可选项；None 一律不放入，避免 CLI 收到 null 报参数错
    optional = {
        "checkIn": check_in,
        "checkOut": check_out,
        "keyword": keyword,
        "prices": prices,
        "queryId": query_id,
        "pageNum": page_num,
    }
    for k, v in optional.items():
        if v is not None:
            args[k] = v

    return await call_tuniu(
        "hotel",
        "tuniu_hotel_search",
        args,
        ttl_sec=_ttl_for("hotel", "tuniu_hotel_search"),
    )


async def hotel_detail(
    hotel_id: Optional[int | str] = None,
    hotel_name: Optional[str] = None,
    check_in: Optional[str] = None,
    check_out: Optional[str] = None,
) -> Any:
    """途牛酒店详情 hotel.tuniu_hotel_detail。

    hotel_id / hotel_name 二选一必填；同时传入时优先 hotel_id（避免歧义）。
    """
    if hotel_id is None and not hotel_name:
        raise ValueError("hotel_detail: hotel_id 与 hotel_name 至少传一个")

    args: dict[str, Any] = {}
    if hotel_id is not None:
        args["hotelId"] = int(hotel_id) if isinstance(hotel_id, str) else hotel_id
    elif hotel_name:
        args["hotelName"] = hotel_name

    if check_in is not None:
        args["checkIn"] = check_in
    if check_out is not None:
        args["checkOut"] = check_out

    return await call_tuniu(
        "hotel",
        "tuniu_hotel_detail",
        args,
        ttl_sec=_ttl_for("hotel", "tuniu_hotel_detail"),
    )


async def flight_search(
    departure_city: str,
    arrival_city: str,
    departure_date: str,
    search_type: Optional[str] = None,
    departure_time: Optional[str] = None,
    arrival_time: Optional[str] = None,
    page_num: Optional[int] = None,
) -> Any:
    """途牛航班低价搜索 flight.searchLowestPriceFlight。

    必填：departure_city / arrival_city / departure_date(YYYY-MM-DD)。
    search_type 取值：TIME / PRICE / NEAR_GO / NEAR_BACK / TRANSFER；不传走默认低价模式。
    翻页只需复用同条件 + page_num。
    """
    if not (departure_city and arrival_city and departure_date):
        raise ValueError(
            "flight_search: departure_city / arrival_city / departure_date 均不能为空"
        )

    args: dict[str, Any] = {
        "departureCityName": departure_city,
        "arrivalCityName": arrival_city,
        "departureDate": departure_date,
    }
    optional = {
        "searchType": search_type,
        "departureTime": departure_time,
        "arrivalTime": arrival_time,
        "pageNum": page_num,
    }
    for k, v in optional.items():
        if v is not None:
            args[k] = v

    return await call_tuniu(
        "flight",
        "searchLowestPriceFlight",
        args,
        ttl_sec=_ttl_for("flight", "searchLowestPriceFlight"),
    )


# ---------------- 字段映射 ----------------
# 把 tuniu CLI 的 MCP 信封 + 原始字段，转成下游消费者能直接用的扁平 dict。
# 命名统一 snake_case，与 accommodation_agent._merge_hotel_data 的目标字段对齐。
# 缺失字段一律填 None，调用方按需兜底，不在此抛异常。


def unwrap_mcp_content(raw: Any) -> Any:
    """剥离 MCP 信封：raw["content"][0]["text"] → json.loads 内层。

    适用于 tuniu hotel/flight 等 CLI 工具的返回。无信封或解析失败时原样返回。
    """
    if not isinstance(raw, dict):
        return raw
    content = raw.get("content")
    if not isinstance(content, list) or not content:
        return raw
    first = content[0]
    if not isinstance(first, dict) or first.get("type") != "text":
        return raw
    text = first.get("text")
    if not isinstance(text, str):
        return raw
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def iter_hotels(unwrapped: Any) -> list[dict]:
    """从 unwrap_mcp_content 结果中取 hotels 列表。结构异常返回 []。"""
    if not isinstance(unwrapped, dict):
        return []
    hotels = unwrapped.get("hotels")
    if not isinstance(hotels, list):
        return []
    return [h for h in hotels if isinstance(h, dict)]


def normalize_hotel(item: dict) -> dict:
    """将 tuniu 单条酒店转为下游统一字段。缺失键填 None。"""
    return {
        "hotel_id": item.get("hotelId"),
        "hotel_name": item.get("hotelName"),
        "lowest_price": item.get("lowestPrice"),
        "star_name": item.get("starName"),
        "score": item.get("commentScore"),
        "address": item.get("address"),
        "business": item.get("business"),
        "brand": item.get("brandName"),
    }


def iter_flights(unwrapped: Any) -> list[dict]:
    """从 unwrap_mcp_content 结果中取 data 列表（航班）。结构异常返回 []。"""
    if not isinstance(unwrapped, dict):
        return []
    data = unwrapped.get("data")
    if not isinstance(data, list):
        return []
    return [f for f in data if isinstance(f, dict)]


def _to_int_price(raw: Any) -> Any:
    """basePrice 是字符串数字（"710"），转 int；失败保留原值。"""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(raw.strip())
        except ValueError:
            return raw
    return raw


def normalize_flight(item: dict) -> dict:
    """将 tuniu 单条航班转为下游统一字段。"""
    return {
        "flight_no": item.get("flightNumber"),
        "airline": item.get("airlineCompany"),
        "dep_time": item.get("departureTime"),
        "arr_time": item.get("arrivalTime"),
        "dep_airport": item.get("departureAirport"),
        "arr_airport": item.get("arrivalAirport"),
        "duration": item.get("totalDuration"),
        "price": _to_int_price(item.get("basePrice")),
        "cabin_class": item.get("cabinClass"),
    }
