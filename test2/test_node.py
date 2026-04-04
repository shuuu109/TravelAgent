import asyncio
import os
import sys
import traceback

# 确保能导入根目录下的模块
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from langchain_core.messages import HumanMessage
from graph.state import TravelGraphState, HardConstraints
from graph.node import extract_hard_constraints, enrich_soft_constraints, validate_rule_constraints
from mcp_clients.amap_client import amap_mcp_session

# 临时读取环境变量，用于在控制台脱敏打印，确认配置是否生效
AMAP_KEY = os.getenv("AMAP_KEY", "1dd13742a147224131022165e14d6d55")

async def test_mcp_connection_only():
    """专门用于深度排查 MCP 连通性的独立诊断函数"""
    print("=========================================")
    print("🔍 [阶段 0] 开始独立排查高德 MCP SSE 网络连接...")
    
    # 脱敏打印 Key，防止填错或环境变量未加载
    masked_key = f"{AMAP_KEY[:6]}...{AMAP_KEY[-4:]}" if len(AMAP_KEY) > 10 else AMAP_KEY
    print(f"🔑 当前加载的 API Key: {masked_key}")
    
    try:
        async with amap_mcp_session() as session:
            print("✅ SSE 底层长连接建立成功！正在拉取高德空间工具列表...")
            tools_response = await session.list_tools()
            
            # 兼容不同版本 MCP SDK 的返回格式
            tools = getattr(tools_response, 'tools', tools_response)
            
            print(f"🎉 成功获取到 {len(tools)} 个工具：")
            for t in tools:
                print(f"  - 🛠️ {t.name}")
            print("=========================================\n")
            return True
            
    except Exception as e:
        print("\n❌ 连接彻底失败！正在拆解 TaskGroup 真实错误根因：")
        
        # 拆解 Python 的 ExceptionGroup / TaskGroup 子异常
        if hasattr(e, 'exceptions'):
            for i, sub_e in enumerate(e.exceptions):
                print(f"  🔴 子异常 {i+1} [{type(sub_e).__name__}]: {sub_e}")
        else:
            print(f"  🔴 常规异常 [{type(e).__name__}]: {e}")
            
        print("\n📜 完整错误堆栈:")
        traceback.print_exc()
        print("=========================================\n")
        return False

async def run_node_tests():
    # 0. 先跑一次纯粹的连通性诊断
    connection_ok = await test_mcp_connection_only()
    if not connection_ok:
        print("⚠️ 连通性测试未通过，后续 Agent 测试可能无法调用工具，但仍将继续执行以测试图逻辑...\n")

    print("🚀 开始测试出行规划节点链...\n")

    test_input = "我明天要从南京去北京出差，大概3个人，当天必须往返赶回来开会。"
    print(f"👤 用户输入: {test_input}\n")
    
    initial_state: TravelGraphState = {
        "messages": [HumanMessage(content=test_input)],
        "hard_constraints": HardConstraints(),
        "soft_constraints": None,
        "rule_violations": [],
        "missing_info": [],
        "current_plan": {}
    }

    # 测试: 硬约束提取
    print("⏳ [Node 1] 正在提取硬约束...")
    update_1 = extract_hard_constraints(initial_state)
    initial_state["hard_constraints"] = update_1.get("hard_constraints")
    initial_state["missing_info"] = update_1.get("missing_info", [])
    print("✅ 硬约束提取完成。\n")

    # 测试: 软约束增强
    print("⏳ [Node 2] 正在加载软约束 (偏好)...")
    update_2 = enrich_soft_constraints(initial_state)
    initial_state["soft_constraints"] = update_2.get("soft_constraints")
    print("✅ 软约束增强完成。\n")

    # 测试: 规则约束校验
    print("⏳ [Node 3] 正在启动高德 MCP 空间推理 Agent 进行规则校验...")
    update_3 = await validate_rule_constraints(initial_state)
    initial_state["rule_violations"] = update_3.get("rule_violations", [])
    
    print("✅ 物理规则校验结果:")
    if not initial_state["rule_violations"]:
        print("  - 🟢 校验通过，未发现时空冲突。")
    else:
        for i, violation in enumerate(initial_state["rule_violations"], 1):
            print(f"  - 🔴 发现冲突 {i}:")
            print(f"    - 类型: {violation.violation_type}")
            print(f"    - 描述: {violation.description}")
            print(f"    - 建议: {violation.suggestion}")

if __name__ == "__main__":
    asyncio.run(run_node_tests())