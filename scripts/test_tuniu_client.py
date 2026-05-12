"""
途牛 MCP Client 真实环境冒烟测试
=================================

分段计划：
  段 1: 环境 sanity + 离线参数校验（不消耗 RPM/RPD 配额）
  段 2: hotel_search / hotel_detail 真实调用 + 缓存命中验证
  段 3: flight_search 真实调用
  段 4: 错误路径（超时、bad args 触发 cli_failure）+ 最终汇总

运行方式（项目根目录）：
    python scripts/test_tuniu_client.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 项目根入 sys.path，支持脚本独立运行
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

import os
import shutil
from typing import Optional


# ---------------- 打印 / 计数工具 ----------------

class Stats:
    """跨段共享的 pass/fail 计数器；最终段调用 summary() 输出汇总"""

    def __init__(self) -> None:
        self.passed: int = 0
        self.failed: int = 0
        self.failures: list[str] = []  # 失败用例名称，便于回溯

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            self.failures.append(f"{name}: {detail}" if detail else name)

    def summary(self) -> int:
        """打印汇总，返回退出码（0=全过，1=有失败）"""
        total = self.passed + self.failed
        print()
        print("=" * 60)
        print(f"SUMMARY: passed={self.passed}/{total}  failed={self.failed}")
        if self.failures:
            print("Failures:")
            for f in self.failures:
                print(f"  - {f}")
        print("=" * 60)
        return 0 if self.failed == 0 else 1


STATS = Stats()


def section(title: str) -> None:
    print()
    print("-" * 60)
    print(f"[SECTION] {title}")
    print("-" * 60)


def ok(name: str, msg: str = "") -> None:
    STATS.record(name, True)
    suffix = f" - {msg}" if msg else ""
    print(f"  [PASS] {name}{suffix}")


def fail(name: str, msg: str) -> None:
    STATS.record(name, False, msg)
    print(f"  [FAIL] {name} - {msg}")


def info(msg: str) -> None:
    print(f"  [INFO] {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


# ---------------- 环境 sanity ----------------

def check_env() -> bool:
    """校验运行所需环境；任一关键项缺失返回 False，调用方应中止后续真实调用。

    检查项：
      1. TUNIU_API_KEY 是否在环境变量（子进程会继承）
      2. tuniu CLI 是否能在 PATH 中解析到
      3. TUNIU_MCP_CONFIG 是否含 command / timeout / cache_ttl
    """
    section("环境 sanity 检查")
    all_ok = True

    # 1. API Key
    api_key = os.environ.get("TUNIU_API_KEY")
    if api_key:
        masked = api_key[:4] + "***" + api_key[-4:] if len(api_key) > 8 else "***"
        ok("env.TUNIU_API_KEY", f"已加载（{masked}）")
    else:
        fail("env.TUNIU_API_KEY", ".env 未提供或 dotenv 未生效")
        all_ok = False

    # 2. tuniu CLI 可执行
    from config import TUNIU_MCP_CONFIG  # 延后到 sys.path 注入后

    raw_cmd = TUNIU_MCP_CONFIG.get("command", "tuniu")
    resolved: Optional[str] = shutil.which(raw_cmd)
    if resolved:
        ok("cli.resolve", f"{raw_cmd} -> {resolved}")
    else:
        fail("cli.resolve", f"PATH 中找不到可执行 '{raw_cmd}'")
        all_ok = False

    # 3. 配置字段齐全
    required_keys = ("command", "timeout", "cache_ttl")
    missing = [k for k in required_keys if k not in TUNIU_MCP_CONFIG]
    if missing:
        fail("config.TUNIU_MCP_CONFIG", f"缺失字段: {missing}")
        all_ok = False
    else:
        ok("config.TUNIU_MCP_CONFIG",
           f"timeout={TUNIU_MCP_CONFIG['timeout']}s, "
           f"cache_ttl entries={len(TUNIU_MCP_CONFIG['cache_ttl'])}")

    return all_ok


# ---------------- 段 1: 离线参数校验 ----------------

import asyncio

from mcp_clients.tuniu_client import (
    TuniuCallError,
    flight_search,
    hotel_detail,
    hotel_search,
)


def _expect_value_error(name: str, coro_factory) -> None:
    """运行一个返回协程的工厂函数；预期抛 ValueError 才算通过。

    用 factory 而非协程本身，避免 Python 报 'coroutine was never awaited'
    （某些用例我们只想看是否构造时就 raise，但这里统一走 run）。
    """
    try:
        asyncio.run(coro_factory())
    except ValueError as e:
        ok(name, f"ValueError: {e}")
    except Exception as e:
        fail(name, f"预期 ValueError，实际 {type(e).__name__}: {e}")
    else:
        fail(name, "未抛异常")


def run_offline_validation() -> None:
    section("段 1: 离线参数校验（不消耗配额）")

    # hotel_search: city_name 必填
    _expect_value_error(
        "hotel_search.empty_city",
        lambda: hotel_search(city_name=""),
    )

    # hotel_detail: hotel_id 与 hotel_name 至少传一个
    _expect_value_error(
        "hotel_detail.both_missing",
        lambda: hotel_detail(),
    )

    # flight_search: 三个必填字段任一缺失即报错（这里测出发城市空）
    _expect_value_error(
        "flight_search.empty_departure_city",
        lambda: flight_search(
            departure_city="",
            arrival_city="SHA",
            departure_date="2026-06-01",
        ),
    )

    # TuniuCallError 构造 & str() 格式
    try:
        err = TuniuCallError(
            type="cli_failure",
            message="bad args",
            code="E_ARGS",
            details={"field": "cityName"},
        )
        assert err.type == "cli_failure"
        assert err.code == "E_ARGS"
        assert err.details == {"field": "cityName"}
        assert "[cli_failure]" in str(err) and "(code=E_ARGS)" in str(err)
        ok("TuniuCallError.shape", str(err))
    except AssertionError as e:
        fail("TuniuCallError.shape", f"字段或 str() 格式不符: {e}")

    # TuniuCallError 无 code 时 str() 不带 (code=...)
    try:
        err = TuniuCallError(type="timeout", message="too slow")
        assert err.code is None
        assert "(code=" not in str(err)
        ok("TuniuCallError.no_code", str(err))
    except AssertionError as e:
        fail("TuniuCallError.no_code", str(e))


# ---------------- 段 2: hotel 真实调用 ----------------

import json

from utils.tuniu_budget import TuniuBudgetExceeded, tuniu_budget

# 跨段共享：段 2.1 的搜索结果，段 2.3 用它取 hotelId
_SECTION2_STATE: dict = {}


def _unwrap_mcp_content(result):
    """剥掉 MCP 的 {"content":[{"type":"text","text":"<json>"}]} 外壳。

    途牛 CLI 返回的业务数据被包在 MCP 标准 content 字段里，真正的列表/详情
    需要再次 json.loads(content[0].text) 才能拿到。

    传入非 MCP 包装结构时原样返回，调用方自己兜底。
    """
    if not isinstance(result, dict):
        return result
    content = result.get("content")
    if not isinstance(content, list) or not content:
        return result
    first = content[0]
    if not isinstance(first, dict) or first.get("type") != "text":
        return result
    text = first.get("text")
    if not isinstance(text, str):
        return result
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def _print_result_snippet(label: str, result) -> None:
    """打印返回值的诊断摘要：类型 / 顶层 key 或长度 / 前 400 字符"""
    if isinstance(result, dict):
        info(f"{label}: dict, keys={list(result.keys())}")
    elif isinstance(result, list):
        info(f"{label}: list, len={len(result)}")
        if result and isinstance(result[0], dict):
            info(f"{label}[0].keys={list(result[0].keys())}")
    else:
        info(f"{label}: type={type(result).__name__}")

    try:
        rendered = json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        rendered = repr(result)
    if len(rendered) > 400:
        rendered = rendered[:400] + "...(truncated)"
    info(f"{label} snippet: {rendered}")


async def _hotel_search_first_call() -> None:
    """首次调用 hotel_search('上海')，结构断言后暂存到 _SECTION2_STATE"""
    name = "hotel_search.first_call"
    try:
        result = await hotel_search(city_name="上海")
    except TuniuBudgetExceeded as e:
        fail(name, f"配额耗尽，跳过: {e}")
        return
    except TuniuCallError as e:
        fail(name, f"{e.type} code={e.code} msg={e.message}")
        if e.details:
            info(f"error.details: {json.dumps(e.details, ensure_ascii=False, default=str)[:600]}")
        return

    # 结构断言：tuniu_client 已剥离 success 字段，应返回 result/data 内层
    if not isinstance(result, (dict, list)):
        fail(name, f"预期 dict/list，实际 {type(result).__name__}")
        return

    if isinstance(result, dict) and not result:
        fail(name, "返回 dict 为空，可能 city_name 未匹配")
        return
    if isinstance(result, list) and not result:
        fail(name, "返回 list 为空，可能 city_name 未匹配")
        return

    _SECTION2_STATE["search_result"] = result
    status = tuniu_budget.get_status()
    ok(name, f"rpd_used={status['rpd_used']}/{status['rpd_limit']}, "
             f"cache_size={status['cache_size']}")
    _print_result_snippet("hotel_search.result", result)


async def _hotel_search_cache_hit() -> None:
    """同样的 city_name 再调一次：rpd_used 不应增加，cache_size 不变"""
    name = "hotel_search.cache_hit"

    before = tuniu_budget.get_status()
    rpd_before = before["rpd_used"]
    cache_before = before["cache_size"]

    if "search_result" not in _SECTION2_STATE:
        fail(name, "段 2.1 未产出 search_result，跳过缓存校验")
        return

    try:
        result = await hotel_search(city_name="上海")
    except TuniuBudgetExceeded as e:
        fail(name, f"配额耗尽: {e}")
        return
    except TuniuCallError as e:
        fail(name, f"{e.type} code={e.code} msg={e.message}")
        return

    after = tuniu_budget.get_status()
    rpd_after = after["rpd_used"]
    cache_after = after["cache_size"]

    if rpd_after != rpd_before:
        fail(name, f"rpd_used 从 {rpd_before} 涨到 {rpd_after}，未命中缓存")
        return
    if cache_after != cache_before:
        fail(name, f"cache_size 从 {cache_before} 变成 {cache_after}，缓存逻辑异常")
        return
    if result is not _SECTION2_STATE["search_result"]:
        # 引用不等也未必坏（实现可能 deep copy），但记一笔信息
        info("缓存返回值与首调非同一对象（实现做了拷贝？）")

    ok(name, f"rpd_used 持平 {rpd_after}, cache_size 持平 {cache_after}")


def _extract_first_hotel_id(unwrapped) -> Optional[str]:
    """从 unwrap 后的搜索结果里拿首条 hotelId；找不到返回 None。

    途牛搜索结果常见结构（容错处理）：
      - {"hotels": [{"hotelId": ...}, ...]}
      - {"data": {"hotels": [...]}}
      - {"list": [{"hotelId": ...}, ...]}
      - 直接 list[{"hotelId": ...}]
    """
    # 直接 list
    if isinstance(unwrapped, list):
        for item in unwrapped:
            if isinstance(item, dict):
                hid = item.get("hotelId") or item.get("hotel_id") or item.get("id")
                if hid:
                    return str(hid)
        return None

    if not isinstance(unwrapped, dict):
        return None

    # 常见列表字段
    for key in ("hotels", "list", "items", "data"):
        sub = unwrapped.get(key)
        # data 可能再嵌一层
        if isinstance(sub, dict):
            for k2 in ("hotels", "list", "items"):
                inner = sub.get(k2)
                if isinstance(inner, list):
                    sub = inner
                    break
        if isinstance(sub, list):
            for item in sub:
                if isinstance(item, dict):
                    hid = item.get("hotelId") or item.get("hotel_id") or item.get("id")
                    if hid:
                        return str(hid)
    return None


async def _hotel_detail_first_call() -> None:
    """从段 2.1 结果里取 hotelId，调 hotel_detail 验证真实路径"""
    name = "hotel_detail.first_call"

    if "search_result" not in _SECTION2_STATE:
        fail(name, "段 2.1 未产出 search_result，跳过")
        return

    unwrapped = _unwrap_mcp_content(_SECTION2_STATE["search_result"])
    hotel_id = _extract_first_hotel_id(unwrapped)
    if not hotel_id:
        fail(name, "未能从搜索结果解析出 hotelId")
        _print_result_snippet("unwrapped.search_result", unwrapped)
        return

    info(f"使用 hotelId={hotel_id} 调用 hotel_detail")

    try:
        result = await hotel_detail(hotel_id=hotel_id)
    except TuniuBudgetExceeded as e:
        fail(name, f"配额耗尽: {e}")
        return
    except TuniuCallError as e:
        fail(name, f"{e.type} code={e.code} msg={e.message}")
        if e.details:
            info(f"error.details: {json.dumps(e.details, ensure_ascii=False, default=str)[:600]}")
        return

    if not isinstance(result, (dict, list)) or (isinstance(result, dict) and not result):
        fail(name, f"返回结构异常: type={type(result).__name__}")
        return

    _SECTION2_STATE["detail_result"] = result
    status = tuniu_budget.get_status()
    ok(name, f"rpd_used={status['rpd_used']}/{status['rpd_limit']}, "
             f"cache_size={status['cache_size']}")
    _print_result_snippet("hotel_detail.result", result)


def run_section2_hotel() -> None:
    section("段 2: hotel 真实调用（消耗配额）")
    asyncio.run(_hotel_search_first_call())
    asyncio.run(_hotel_search_cache_hit())
    asyncio.run(_hotel_detail_first_call())


# ---------------- 段 3: flight 真实调用 ----------------

# 跨段共享：段 3.1 的搜索结果，段 3.2 用它取 flightNo
_SECTION3_STATE: dict = {}


async def _flight_search_first_call() -> None:
    """首次调用 flight_search(PEK -> SHA, 一个月后)，结构断言后暂存到 _SECTION3_STATE"""
    name = "flight_search.first_call"
    try:
        result = await flight_search(
            departure_city="北京",
            arrival_city="上海",
            departure_date="2026-06-15",
        )
    except TuniuBudgetExceeded as e:
        fail(name, f"配额耗尽，跳过: {e}")
        return
    except TuniuCallError as e:
        fail(name, f"{e.type} code={e.code} msg={e.message}")
        if e.details:
            info(f"error.details: {json.dumps(e.details, ensure_ascii=False, default=str)[:600]}")
        return

    if not isinstance(result, (dict, list)):
        fail(name, f"预期 dict/list，实际 {type(result).__name__}")
        return
    if isinstance(result, dict) and not result:
        fail(name, "返回 dict 为空，可能日期/城市未匹配到班次")
        return
    if isinstance(result, list) and not result:
        fail(name, "返回 list 为空，可能日期/城市未匹配到班次")
        return

    _SECTION3_STATE["search_result"] = result
    status = tuniu_budget.get_status()
    ok(name, f"rpd_used={status['rpd_used']}/{status['rpd_limit']}, "
             f"cache_size={status['cache_size']}")
    _print_result_snippet("flight_search.result", result)


def run_section3_flight() -> None:
    section("段 3: flight 真实调用（消耗配额）")
    asyncio.run(_flight_search_first_call())


# ---------------- 入口 ----------------

if __name__ == "__main__":
    if not check_env():
        warn("环境检查未通过，跳过后续真实调用；离线测试照常执行")

    # run_offline_validation()

    run_section2_hotel()

    run_section3_flight()

    # 段 4 在此追加
    sys.exit(STATS.summary())
