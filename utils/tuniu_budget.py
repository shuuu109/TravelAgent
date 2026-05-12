"""
途牛 MCP 调用预算管控

职责：
1. RPM 限流：每分钟最多 5 次调用，超限短等待；等不到则抛 TuniuBudgetExceeded
2. RPD 配额：每自然日（Asia/Shanghai, UTC+8）最多 50 次调用，超限直接抛错
3. 结果缓存：按 (domain, tool, args) 缓存返回值，TTL 由调用方指定

设计取舍：
- 进程内单例，不做文件持久化（毕设单进程场景，重启重置可接受）
- 限流仅短等待，不做指数退避（窗口固定，等也是等到下一秒）
- 所有计数与缓存在同一把 asyncio.Lock 下，简单可靠
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# 自然日基准时区：Asia/Shanghai = UTC+8
_TZ_SHANGHAI = timezone(timedelta(hours=8))

# 默认配额（Commit 2 接入 config.py 后由调用方覆盖）
DEFAULT_RPM = 5
DEFAULT_RPD = 50

# 限流等待上限：超过则放弃等待并抛错（避免阻塞 LangGraph）
DEFAULT_MAX_WAIT_SEC = 12.0


class TuniuBudgetExceeded(Exception):
    """途牛调用预算耗尽（RPM 等待超时 或 RPD 已用满）"""
    pass


class TuniuBudget:
    """途牛 MCP 调用的限流 + 配额 + 缓存管控器（进程内单例）"""

    def __init__(
        self,
        rpm: int = DEFAULT_RPM,
        rpd: int = DEFAULT_RPD,
        max_wait_sec: float = DEFAULT_MAX_WAIT_SEC,
    ):
        """
        Args:
            rpm: 每分钟最大请求数（滑动窗口）
            rpd: 每自然日最大请求数（Asia/Shanghai 跨日重置）
            max_wait_sec: RPM 受限时最长等待秒数，超时抛 TuniuBudgetExceeded
        """
        self.rpm = rpm
        self.rpd = rpd
        self.max_wait_sec = max_wait_sec

        # RPM 滑动窗口：最近一分钟内的请求时间戳（time.monotonic）
        self._minute_window: deque[float] = deque()

        # RPD 计数：当前自然日 + 已消费次数
        self._day_key: str = self._current_day_key()
        self._day_count: int = 0

        # 结果缓存：key -> (value, expires_at_monotonic)
        self._cache: dict[str, tuple[Any, float]] = {}

        # 全局锁：保护 _minute_window / _day_* / _cache
        self._lock = asyncio.Lock()

    @staticmethod
    def _current_day_key() -> str:
        """返回当前 Asia/Shanghai 自然日的字符串键，例如 '2026-05-12'"""
        return datetime.now(_TZ_SHANGHAI).strftime("%Y-%m-%d")

    async def acquire(self) -> None:
        """
        申请一次调用配额。
        - RPD 未满 + RPM 未满 → 立即记账返回
        - RPM 已满但等待可在 max_wait_sec 内解除 → 短睡后重试
        - RPD 已满，或 RPM 等待超时 → 抛 TuniuBudgetExceeded
        """
        total_waited = 0.0

        while True:
            async with self._lock:
                # 跨日重置 RPD
                today = self._current_day_key()
                if today != self._day_key:
                    logger.info(
                        "Tuniu budget: day rollover %s -> %s, reset RPD count",
                        self._day_key, today,
                    )
                    self._day_key = today
                    self._day_count = 0

                # RPD 优先校验：满了没救
                if self._day_count >= self.rpd:
                    raise TuniuBudgetExceeded(
                        f"途牛今日调用已达上限 ({self._day_count}/{self.rpd})，请明日再试"
                    )

                # 清理 RPM 窗口：剔除 60 秒前的时间戳
                now = time.monotonic()
                cutoff = now - 60.0
                while self._minute_window and self._minute_window[0] <= cutoff:
                    self._minute_window.popleft()

                # RPM 未满：直接记账返回
                if len(self._minute_window) < self.rpm:
                    self._minute_window.append(now)
                    self._day_count += 1
                    return

                # RPM 已满：算出窗口最早一次调用还需多久过期
                wait_needed = 60.0 - (now - self._minute_window[0])
                # +0.05 缓冲，避免边界抖动后再回锁仍卡在同一窗口
                wait_needed = max(wait_needed, 0.0) + 0.05

            # 锁外等待，给其他协程读缓存的机会
            if total_waited + wait_needed > self.max_wait_sec:
                raise TuniuBudgetExceeded(
                    f"途牛 RPM 限流等待超过 {self.max_wait_sec}s，放弃本次调用"
                )

            logger.debug("Tuniu budget: RPM full, sleep %.2fs before retry", wait_needed)
            await asyncio.sleep(wait_needed)
            total_waited += wait_needed

    @staticmethod
    def make_cache_key(domain: str, tool: str, args: Optional[dict]) -> str:
        """
        生成稳定缓存键：domain:tool:sha1(args_json)[:16]
        args=None 视同 {}；嵌套 dict/list 通过 sort_keys 保证顺序一致
        """
        payload = json.dumps(
            args or {},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
        return f"{domain}:{tool}:{digest}"

    def cache_get(self, key: str) -> Optional[Any]:
        """命中返回值；未命中或已过期返回 None（顺手淘汰过期项）"""
        item = self._cache.get(key)
        if item is None:
            return None
        value, expires_at = item
        if time.monotonic() >= expires_at:
            self._cache.pop(key, None)
            return None
        return value

    def cache_set(self, key: str, value: Any, ttl_sec: float) -> None:
        """写入缓存；ttl_sec<=0 视为不缓存（直接忽略）"""
        if ttl_sec <= 0:
            return
        self._cache[key] = (value, time.monotonic() + ttl_sec)

    def get_status(self) -> dict:
        """状态摘要，给 /status 命令或日志用"""
        # 同步快照：剔一次过期窗口让 rpm_used 更准
        now = time.monotonic()
        cutoff = now - 60.0
        while self._minute_window and self._minute_window[0] <= cutoff:
            self._minute_window.popleft()
        return {
            "rpm_used": len(self._minute_window),
            "rpm_limit": self.rpm,
            "rpd_used": self._day_count,
            "rpd_limit": self.rpd,
            "day_key": self._day_key,
            "cache_size": len(self._cache),
        }


# 模块级单例：所有 tuniu MCP 调用共享同一份配额
tuniu_budget = TuniuBudget()


def with_budget(ttl_sec: float = 0):
    """
    装饰器：把 `async fn(domain, tool, args=None, ...)` 包成"先查缓存 → 再过限流 → 落缓存"。

    Args:
        ttl_sec: 命中后的缓存有效期（秒）；<=0 表示不缓存（仅走限流）

    用法：
        @with_budget(ttl_sec=300)
        async def call_tuniu(domain, tool, args=None):
            ...
    """
    import functools

    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(domain: str, tool: str, args: Optional[dict] = None, *a, **kw):
            key = TuniuBudget.make_cache_key(domain, tool, args)
            cached = tuniu_budget.cache_get(key)
            if cached is not None:
                logger.debug("Tuniu cache hit: %s", key)
                return cached

            await tuniu_budget.acquire()
            result = await fn(domain, tool, args, *a, **kw)
            tuniu_budget.cache_set(key, result, ttl_sec)
            return result

        return wrapper

    return deco
