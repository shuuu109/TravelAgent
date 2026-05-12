"""
高德 maps_weather 原始返回诊断脚本
=================================

目的：定位 TransportAgent 中 weather_summary 为空串的根因。
只读，不修改 amap_client.py。

行为：
  - 直接调 get_city_weather(session, "上海"), 打印原始返回 json
  - 再喂给 summarize_weather, 打印返回值
  - 不做 assert, 只供人工核对结构

运行方式（项目根目录）：
    python scripts/test_amap_weather.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")

# 项目根入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from mcp_clients.amap_client import (
    amap_mcp_session,
    get_city_weather,
    summarize_weather,
)


def _hr(title: str = "") -> None:
    print()
    print("-" * 60)
    if title:
        print(f"[SECTION] {title}")
        print("-" * 60)


async def _run(city: str) -> None:
    _hr(f"amap_mcp_session -> get_city_weather(city={city!r})")
    async with amap_mcp_session() as session:
        data = await get_city_weather(session, city)

    _hr("原始返回 (json.dumps, ensure_ascii=False, indent=2)")
    print(f"  type: {type(data).__name__}")
    print(f"  top-level keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
    try:
        print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    except Exception as e:
        print(f"  [WARN] json.dumps 失败: {e}; repr: {data!r}")

    _hr("summarize_weather 解析结果")
    summary = summarize_weather(data)
    print(f"  summary={summary!r} (len={len(summary)})")

    # 额外手动检查解析路径，便于定位 summarize 在哪一步降级
    if isinstance(data, dict):
        forecasts = data.get("forecasts")
        lives = data.get("lives")
        print(f"  data.get('forecasts'): type={type(forecasts).__name__} len="
              f"{len(forecasts) if hasattr(forecasts, '__len__') else 'N/A'}")
        print(f"  data.get('lives'):     type={type(lives).__name__} len="
              f"{len(lives) if hasattr(lives, '__len__') else 'N/A'}")
        if isinstance(forecasts, list) and forecasts:
            first = forecasts[0] or {}
            print(f"  forecasts[0] keys: {list(first.keys()) if isinstance(first, dict) else type(first).__name__}")
            casts = first.get("casts") if isinstance(first, dict) else None
            print(f"  forecasts[0].casts: type={type(casts).__name__} len="
                  f"{len(casts) if hasattr(casts, '__len__') else 'N/A'}")
            if isinstance(casts, list) and casts:
                print(f"  forecasts[0].casts[0]: {casts[0]!r}")


def main() -> int:
    asyncio.run(_run("上海"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
