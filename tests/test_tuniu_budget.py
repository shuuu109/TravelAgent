"""
utils/tuniu_budget.py 单元测试

运行：
    python tests/test_tuniu_budget.py

无 pytest 依赖：用 asyncio.run + assert 跑，每段独立一个 async 函数。
失败时抛 AssertionError，main 汇总通过 / 失败计数。
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback

# 让脚本可独立运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tuniu_budget import (
    TuniuBudget,
    TuniuBudgetExceeded,
    with_budget,
)


# ---------- 缓存相关测试 ----------

async def test_cache_key_stable():
    """同样的 args（不论 dict 顺序）应得到相同 key；args=None 等同于 {}"""
    k1 = TuniuBudget.make_cache_key("hotel", "search", {"city": "北京", "nights": 2})
    k2 = TuniuBudget.make_cache_key("hotel", "search", {"nights": 2, "city": "北京"})
    assert k1 == k2, f"dict 顺序不应影响 key: {k1} vs {k2}"

    k_none = TuniuBudget.make_cache_key("hotel", "search", None)
    k_empty = TuniuBudget.make_cache_key("hotel", "search", {})
    assert k_none == k_empty, "args=None 应等同于空 dict"

    # 不同 domain/tool/args 必须区分
    assert TuniuBudget.make_cache_key("hotel", "search", {"a": 1}) != \
           TuniuBudget.make_cache_key("flight", "search", {"a": 1})
    assert TuniuBudget.make_cache_key("hotel", "search", {"a": 1}) != \
           TuniuBudget.make_cache_key("hotel", "detail", {"a": 1})
    assert TuniuBudget.make_cache_key("hotel", "search", {"a": 1}) != \
           TuniuBudget.make_cache_key("hotel", "search", {"a": 2})


async def test_cache_set_get_hit():
    """set 后 get 能命中且返回原值"""
    b = TuniuBudget()
    key = "hotel:search:abc"
    b.cache_set(key, {"hotels": [1, 2, 3]}, ttl_sec=10)
    got = b.cache_get(key)
    assert got == {"hotels": [1, 2, 3]}, f"期待命中原值，实际 {got}"


async def test_cache_ttl_expire():
    """TTL 过期后 get 返回 None，并顺手淘汰"""
    b = TuniuBudget()
    key = "k_expire"
    b.cache_set(key, "value", ttl_sec=0.05)
    assert b.cache_get(key) == "value", "刚 set 应能命中"
    await asyncio.sleep(0.1)
    assert b.cache_get(key) is None, "TTL 过期后应返回 None"
    assert key not in b._cache, "过期项应被淘汰"


async def test_cache_ttl_zero_skip():
    """ttl_sec <= 0 不写入缓存"""
    b = TuniuBudget()
    b.cache_set("k_zero", "v", ttl_sec=0)
    b.cache_set("k_neg", "v", ttl_sec=-1)
    assert b.cache_get("k_zero") is None
    assert b.cache_get("k_neg") is None
    assert len(b._cache) == 0


# ---------- acquire 限流测试 ----------

async def test_acquire_normal_under_limit():
    """RPM/RPD 都没满时，acquire 立即返回且计数递增"""
    b = TuniuBudget(rpm=5, rpd=50)
    for i in range(3):
        t0 = time.monotonic()
        await b.acquire()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.05, f"第 {i+1} 次 acquire 不应等待，实际 {elapsed:.3f}s"

    status = b.get_status()
    assert status["rpm_used"] == 3, f"rpm_used 应为 3，实际 {status['rpm_used']}"
    assert status["rpd_used"] == 3, f"rpd_used 应为 3，实际 {status['rpd_used']}"


async def test_acquire_window_slides():
    """60s 前的旧时间戳应被滑动窗口剔除，不占 RPM 名额"""
    b = TuniuBudget(rpm=2, rpd=50)
    # 手动塞两个 70 秒前的"陈旧"时间戳
    old = time.monotonic() - 70.0
    b._minute_window.append(old)
    b._minute_window.append(old)
    # _day_count 也手动同步一下，保持账面真实
    b._day_count = 2

    t0 = time.monotonic()
    await b.acquire()  # 应当瞬间通过：旧窗口已过期
    elapsed = time.monotonic() - t0
    assert elapsed < 0.05, f"过期窗口应被清理，不应等待，实际 {elapsed:.3f}s"

    status = b.get_status()
    assert status["rpm_used"] == 1, f"清理后只剩本次，rpm_used 应为 1，实际 {status['rpm_used']}"


async def test_acquire_rpm_short_wait_success():
    """RPM 已满但等待可在 max_wait_sec 内解除：短睡后成功 acquire"""
    b = TuniuBudget(rpm=2, rpd=50, max_wait_sec=2.0)
    # 塞两个 "59.8 秒前" 的时间戳，意味着第三次 acquire 需要等 ~0.2s
    near_expire = time.monotonic() - 59.8
    b._minute_window.append(near_expire)
    b._minute_window.append(near_expire)
    b._day_count = 2

    t0 = time.monotonic()
    await b.acquire()  # 应在 ~0.25s 后通过
    elapsed = time.monotonic() - t0
    assert 0.1 < elapsed < 1.0, f"应短等待 ~0.2s，实际 {elapsed:.3f}s"

    status = b.get_status()
    # 旧两条已过期 + 本次新增 = 1
    assert status["rpm_used"] == 1, f"窗口里应只剩本次，rpm_used 应为 1，实际 {status['rpm_used']}"


# ---------- 失败路径测试 ----------

async def test_acquire_rpm_wait_timeout():
    """RPM 已满且最早一条还要等很久 → 抛 TuniuBudgetExceeded"""
    b = TuniuBudget(rpm=2, rpd=50, max_wait_sec=0.1)
    # 塞两个 "刚才" 的时间戳——离过期还有 ~60s，远大于 max_wait_sec=0.1
    now = time.monotonic()
    b._minute_window.append(now)
    b._minute_window.append(now)
    b._day_count = 2

    raised = False
    t0 = time.monotonic()
    try:
        await b.acquire()
    except TuniuBudgetExceeded as e:
        raised = True
        msg = str(e)
        assert "RPM" in msg or "限流" in msg, f"异常消息应提示 RPM 限流，实际：{msg}"
    elapsed = time.monotonic() - t0

    assert raised, "RPM 等待超过 max_wait_sec 应抛 TuniuBudgetExceeded"
    # 等待时间不应远超 max_wait_sec（一次 sleep + 抛错）
    assert elapsed < 1.0, f"超时退出应快，实际 {elapsed:.3f}s"


async def test_acquire_rpd_exhausted():
    """RPD 已用满 → 立刻抛 TuniuBudgetExceeded，不进入 RPM 等待"""
    b = TuniuBudget(rpm=5, rpd=3, max_wait_sec=5.0)
    # 先正常用满 3 次
    for _ in range(3):
        await b.acquire()

    raised = False
    t0 = time.monotonic()
    try:
        await b.acquire()
    except TuniuBudgetExceeded as e:
        raised = True
        msg = str(e)
        assert "今日" in msg or "RPD" in msg or "上限" in msg, \
            f"异常消息应提示 RPD 上限，实际：{msg}"
    elapsed = time.monotonic() - t0

    assert raised, "RPD 用满应抛 TuniuBudgetExceeded"
    assert elapsed < 0.05, f"RPD 满应立即抛错，不应等待，实际 {elapsed:.3f}s"

    # 状态应保持在上限，不应继续累加
    status = b.get_status()
    assert status["rpd_used"] == 3, f"失败的 acquire 不应增加计数，实际 {status['rpd_used']}"


# ---------- 跨日重置 ----------

async def test_day_rollover_resets_rpd():
    """day_key 变化时，RPD 计数应重置为 0"""
    b = TuniuBudget(rpm=5, rpd=3)
    # 先用满
    for _ in range(3):
        await b.acquire()
    assert b.get_status()["rpd_used"] == 3

    # 模拟到了第二天：直接改 _day_key 让 acquire 内部检测出跨日
    b._day_key = "1999-01-01"

    await b.acquire()  # 应触发 rollover，重置后第 1 次
    status = b.get_status()
    assert status["rpd_used"] == 1, f"跨日重置后应为 1，实际 {status['rpd_used']}"
    assert status["day_key"] != "1999-01-01", "day_key 应已更新为今日"


# ---------- with_budget 装饰器 ----------

async def test_with_budget_cache_hit_skips_call():
    """缓存命中时不应调用底层 fn，也不消耗 acquire 配额"""
    # 用独立 budget 隔离全局单例
    from utils import tuniu_budget as mod
    saved = mod.tuniu_budget
    mod.tuniu_budget = TuniuBudget(rpm=5, rpd=50)

    try:
        call_count = 0

        @with_budget(ttl_sec=60)
        async def fake_call(domain, tool, args=None):
            nonlocal call_count
            call_count += 1
            return {"echo": args}

        r1 = await fake_call("hotel", "search", {"city": "北京"})
        r2 = await fake_call("hotel", "search", {"city": "北京"})

        assert r1 == r2 == {"echo": {"city": "北京"}}, f"返回值应一致：{r1} / {r2}"
        assert call_count == 1, f"第二次应命中缓存，fn 只调用 1 次，实际 {call_count}"

        # 只消耗了 1 次配额
        assert mod.tuniu_budget.get_status()["rpd_used"] == 1
    finally:
        mod.tuniu_budget = saved


async def test_with_budget_miss_writes_cache():
    """缓存未命中时：走 acquire → 调 fn → 写缓存"""
    from utils import tuniu_budget as mod
    saved = mod.tuniu_budget
    mod.tuniu_budget = TuniuBudget(rpm=5, rpd=50)

    try:
        @with_budget(ttl_sec=60)
        async def fake_call(domain, tool, args=None):
            return {"hit": True}

        # 不同 args 不应共享缓存
        await fake_call("hotel", "search", {"city": "A"})
        await fake_call("hotel", "search", {"city": "B"})

        status = mod.tuniu_budget.get_status()
        assert status["rpd_used"] == 2, f"两次不同 args 各消耗一次，实际 {status['rpd_used']}"
        assert status["cache_size"] == 2, f"应有 2 条缓存项，实际 {status['cache_size']}"
    finally:
        mod.tuniu_budget = saved


async def test_with_budget_ttl_zero_no_cache():
    """ttl_sec=0 时不写缓存，每次都走 fn 与 acquire"""
    from utils import tuniu_budget as mod
    saved = mod.tuniu_budget
    mod.tuniu_budget = TuniuBudget(rpm=5, rpd=50)

    try:
        call_count = 0

        @with_budget(ttl_sec=0)
        async def fake_call(domain, tool, args=None):
            nonlocal call_count
            call_count += 1
            return call_count

        r1 = await fake_call("hotel", "detail", {"id": 1})
        r2 = await fake_call("hotel", "detail", {"id": 1})

        assert call_count == 2, f"ttl=0 不应缓存，fn 应被调 2 次，实际 {call_count}"
        assert r1 == 1 and r2 == 2, "两次返回值应不同"
        assert mod.tuniu_budget.get_status()["cache_size"] == 0, "不应写入缓存"
    finally:
        mod.tuniu_budget = saved


# ---------- 运行框架 ----------

_results: list[tuple[str, bool, str]] = []


async def _run(name: str, coro_fn) -> None:
    """跑一个测试用例，捕获 AssertionError / 其他异常并记录"""
    try:
        await coro_fn()
    except AssertionError as e:
        _results.append((name, False, f"AssertionError: {e}"))
        print(f"[FAIL] {name}: {e}")
        return
    except Exception as e:
        _results.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"[ERROR] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return
    _results.append((name, True, ""))
    print(f"[PASS] {name}")


def _summary() -> int:
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = len(_results) - passed
    print("\n" + "=" * 60)
    print(f"  Total: {len(_results)}  Passed: {passed}  Failed: {failed}")
    print("=" * 60)
    return 0 if failed == 0 else 1


async def main() -> int:
    # 缓存相关
    await _run("cache_key_stable", test_cache_key_stable)
    await _run("cache_set_get_hit", test_cache_set_get_hit)
    await _run("cache_ttl_expire", test_cache_ttl_expire)
    await _run("cache_ttl_zero_skip", test_cache_ttl_zero_skip)

    # acquire 限流
    await _run("acquire_normal_under_limit", test_acquire_normal_under_limit)
    await _run("acquire_window_slides", test_acquire_window_slides)
    await _run("acquire_rpm_short_wait_success", test_acquire_rpm_short_wait_success)

    # 失败路径
    await _run("acquire_rpm_wait_timeout", test_acquire_rpm_wait_timeout)
    await _run("acquire_rpd_exhausted", test_acquire_rpd_exhausted)

    # 跨日 + 装饰器
    await _run("day_rollover_resets_rpd", test_day_rollover_resets_rpd)
    await _run("with_budget_cache_hit_skips_call", test_with_budget_cache_hit_skips_call)
    await _run("with_budget_miss_writes_cache", test_with_budget_miss_writes_cache)
    await _run("with_budget_ttl_zero_no_cache", test_with_budget_ttl_zero_no_cache)

    return _summary()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
