#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用户场景端到端真实测试 (不使用 Mock)
用法: python tests/test_user_scenario.py
"""
import sys
import os
import asyncio
import json
import nest_asyncio

# 允许事件循环嵌套，解决 "This event loop is already running" 问题
nest_asyncio.apply()

# 设置模块导入路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# 兼容旧路径引入 InformationQueryAgent
sys.path.insert(0, os.path.join(project_root, ".claude", "skills", "query-info", "script"))

async def main():
    from config_agentscope import init_agentscope
    from config import LLM_CONFIG
    from agentscope.model import OpenAIChatModel
    from agents.transport_agent import TransportAgent
    from agent import InformationQueryAgent
    from agentscope.message import Msg

    init_agentscope()
    # 初始化真实大模型
    model = OpenAIChatModel(
        model_name=LLM_CONFIG["model_name"],
        api_key=LLM_CONFIG["api_key"],
        client_kwargs={"base_url": LLM_CONFIG["base_url"]},
        temperature=0.3,
        max_tokens=4096,
        stream=False
    )

    print("\n" + "="*70)
    print("🧑‍💻 模拟用户提问: \"帮我查一下明天从南京到上海的高铁，以及上海的天气！\"")
    print("="*70 + "\n")

    # ---------------------------------------------------------
    # 1. 触发交通智能体 (测试 12306 MCP)
    # ---------------------------------------------------------
    print("🚂 [智能体路由] -> 交通管家 (TransportAgent) 正在通过 MCP 查票并思考...")
    transport_agent = TransportAgent(name="TA", model=model)
    
    # 模拟 Orchestrator 已经提取出结构化实体（没有传 date，测试它的自动补全能力）
    ta_payload = {"context": {"key_entities": {"origin": "南京", "destination": "上海", "date": ""}}}
    
    ta_res = await transport_agent.reply(Msg(name="Orch", content=json.dumps(ta_payload), role="user"))
    
    print("\n✅ TransportAgent 最终返回 (已提取最优方案):")
    try:
        # 尝试美化打印 JSON
        print(json.dumps(json.loads(ta_res.content), indent=2, ensure_ascii=False))
    except:
        print(ta_res.content)

    print("\n" + "-"*70 + "\n")

    # ---------------------------------------------------------
    # 2. 触发信息智能体 (测试 Flight MCP 的天气工具)
    # ---------------------------------------------------------
    print("🌤️ [智能体路由] -> 信息管家 (InformationQueryAgent) 正在通过 MCP 查天气并思考...")
    weather_agent = InformationQueryAgent(name="IQA", model=model)
    
    # 这里直接传自然语言，测试它能否自动路由给 flight_client 的 weather 工具
    iqa_res = await weather_agent.reply(Msg(name="User", content="查一下上海明天的天气", role="user"))
    
    print("\n✅ InformationQueryAgent 最终返回:")
    try:
        print(json.dumps(json.loads(iqa_res.content), indent=2, ensure_ascii=False))
    except:
        print(iqa_res.content)

    print("\n" + "="*70)
    print("🎉 测试完毕！如果上方输出了真实的 12306 推荐车次和真实温度，说明系统已经完全跑通！")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())