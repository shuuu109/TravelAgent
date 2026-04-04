# test_workflow.py (项目根目录下)
import asyncio
from langchain_core.messages import HumanMessage
from graph.workflow import app

async def test_run():
    # 配置多轮对话的唯一 ID
    config = {"configurable": {"thread_id": "test_session_001"}}
    
    print("\n--- 第一轮：提出不合理需求 ---")
    inputs1 = {"messages": [HumanMessage(content="我明天从南京去北京，后天回来。我要走路去。")]}
    async for event in app.astream(inputs1, config, stream_mode="values"):
        if "messages" in event:
            last_msg = event["messages"][-1]
            if last_msg.type == "ai":
                print(f"🤖 助手: {last_msg.content}")

    print("\n--- 第二轮：接受协商，修改需求 ---")
    inputs2 = {"messages": [HumanMessage(content="好吧，那我坐高铁去吧。")]}
    async for event in app.astream(inputs2, config, stream_mode="values"):
        if "messages" in event:
            last_msg = event["messages"][-1]
            if last_msg.type == "ai":
                print(f"🤖 助手: {last_msg.content}")

if __name__ == "__main__":
    asyncio.run(test_run())