import asyncio
import json
import agentscope
from agentscope.message import Msg
import sys
import os

# 将项目根目录 (travelAgent) 添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 现在可以正常导入了
from agents.transport_agent import TransportAgent

async def main():
    # 1. 初始化 AgentScope (请根据你项目的实际情况修改模型配置项)
    # 如果你通常在 config_agentscope.py 中初始化，这里也可以直接引入你的初始化逻辑
    # agentscope.init(model_configs="你的模型配置文件路径.json") 
    
    # 假设你的大模型配置名称叫 "my_model"（请替换为你在 AgentScope 中实际使用的 model_config_name）
    # 如果你在 TransportAgent 内部自己处理了 model 实例化，这里可以传 None 或对应的名称
    agent = TransportAgent(name="TransportAgent", model="my_model")
    
    # 2. 模拟 Orchestrator 传递过来的上下文消息
    mock_input_data = {
        "context": {
            "key_entities": {
                "origin": "北京",
                "destination": "上海",
                "date": "2026-05-01"  # 可以换成明天的日期测试实时查询
            }
        }
    }
    
    # 将字典转为 JSON 字符串塞入 Msg
    input_msg = Msg(
        name="Orchestrator", 
        content=json.dumps(mock_input_data, ensure_ascii=False), 
        role="assistant"
    )
    
    print(f"正在查询 {mock_input_data['context']['key_entities']['origin']} 到 {mock_input_data['context']['key_entities']['destination']} 的交通方案，请稍候...")
    
    # 3. 调用 Agent
    response = await agent.reply(input_msg)
    
    # 4. 打印结果
    print("\n" + "="*50)
    print("=== TransportAgent 返回结果 ===")
    print("="*50)
    
    try:
        # 尝试美化打印 JSON
        parsed_response = json.loads(response.content)
        print(json.dumps(parsed_response, ensure_ascii=False, indent=2))
    except Exception:
        # 如果不是标准 JSON，直接打印纯文本
        print(response.content)

if __name__ == "__main__":
    asyncio.run(main())