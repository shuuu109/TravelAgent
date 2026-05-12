"""
TransportAgent 真实联网烟雾测试
================================

用例：北京 -> 上海, date=明天

行为：
  - 不做 assert，只在终端打印关键字段供人工核对
  - 唯一兜底校验：飞机 / 火车至少一项 options 非空，否则视为整体失败

运行方式（项目根目录）：
    python scripts/test_transport_agent_tuniu.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")

# 项目根入 sys.path，支持脚本独立运行
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from agents.transport_agent import TransportAgent


def _hr(title: str = "") -> None:
    print()
    print("-" * 60)
    if title:
        print(f"[SECTION] {title}")
        print("-" * 60)


def _dump_options(options: list[dict]) -> None:
    """单条一行打印：transport_type / no / dep-arr / price_range / is_recommended"""
    for i, opt in enumerate(options, 1):
        flag = "*" if opt.get("is_recommended") else " "
        print(
            f"  [{flag}] #{i:02d} {opt.get('transport_type'):<3} "
            f"{(opt.get('transport_no') or '-'):<10} "
            f"{(opt.get('departure_time') or '-'):<5}->"
            f"{(opt.get('arrival_time') or '-'):<5} "
            f"{(opt.get('departure_hub') or '-')}->"
            f"{(opt.get('arrival_hub') or '-')}  "
            f"price={opt.get('price_range') or '-'}"
        )


async def _run_case() -> dict:
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    payload = {
        "context": {
            "key_entities": {
                "origin": "北京",
                "destination": "上海",
                "date": tomorrow,
            }
        }
    }
    _hr(f"调用 TransportAgent.run(origin=北京, destination=上海, date={tomorrow})")
    agent = TransportAgent()
    return await agent.run(payload)


def main() -> int:
    result = asyncio.run(_run_case())

    if "error" in result:
        print(f"  [FAIL] agent 返回 error: {result['error']}")
        return 1

    plan = result.get("transport_plan") or {}
    options: list[dict] = plan.get("options") or []
    recommendation = plan.get("recommendation") or {}
    weather = plan.get("weather_summary") or ""
    query_info = plan.get("query_info") or {}

    # 1) options 概览
    _hr("Options 概览")
    print(f"  总数: {len(options)}")

    type_counter = Counter(o.get("transport_type") for o in options)
    print(f"  各 transport_type 计数: {dict(type_counter)}")

    rec_counter = Counter(
        f"{o.get('transport_type')}:{'Y' if o.get('is_recommended') else 'N'}"
        for o in options
    )
    print(f"  各 is_recommended 标记: {dict(rec_counter)}")

    # 2) 单条明细
    _hr("Options 明细（* = is_recommended）")
    if options:
        _dump_options(options)
    else:
        print("  (空)")

    # 3) recommendation / weather / query_info
    _hr("Recommendation / Weather / QueryInfo")
    print(f"  recommendation: {json.dumps(recommendation, ensure_ascii=False)}")
    print(f"  weather_summary: {weather!r}")
    print(f"  query_info: {json.dumps(query_info, ensure_ascii=False)}")

    # 4) 唯一兜底校验
    _hr("Smoke check")
    has_flight = type_counter.get("飞机", 0) > 0
    has_train = type_counter.get("火车", 0) > 0
    if has_flight or has_train:
        print(f"  [PASS] 飞机/火车至少有一项非空 (flight={has_flight}, train={has_train})")
        return 0
    else:
        print("  [FAIL] 飞机和火车 options 都是空的")
        return 1


if __name__ == "__main__":
    sys.exit(main())
