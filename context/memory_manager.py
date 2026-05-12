"""
记忆管理器 (Memory Manager)

短期会话消息已由 LangGraph checkpointer (按 thread_id) 统一管理，
本模块只负责长期记忆（用户偏好、行程历史、跨会话聊天记录）。
"""
from typing import Dict, Any
from .long_term_memory import LongTermMemory
import logging

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    记忆管理器：仅管理长期记忆（跨会话持久化）。
    短期会话上下文请从 graph state（checkpointer）读取。
    """

    def __init__(self, user_id: str, session_id: str, storage_path: str = "data/memory", llm_model=None):
        """
        Args:
            user_id: 用户ID
            session_id: 会话ID（同时也用作 LangGraph 的 thread_id）
            storage_path: 长期记忆存储路径
            llm_model: LLM 模型实例（用于总结长期记忆）
        """
        self.user_id = user_id
        self.session_id = session_id
        self.llm_model = llm_model

        self.long_term = LongTermMemory(user_id, storage_path)

        logger.info(f"Memory manager initialized for user {user_id}, session {session_id}")

    def add_message(self, role: str, content: str, metadata: Dict = None):
        """
        将一条消息写入长期记忆。
        短期会话历史由 graph checkpointer 自动持久化，调用方无需再写。
        """
        # metadata 暂未使用，保留参数以兼容历史调用点
        _ = metadata
        self.long_term.add_chat_message(role, content, self.session_id)

    def end_session(self):
        """结束会话：长期记忆已落盘，短期消息随 thread_id 自然失效，仅记日志。"""
        logger.info(f"Session ended: {self.session_id}")

    async def get_long_term_summary_async(self, max_messages: int = 50) -> str:
        """使用 LLM 总结跨会话历史聊天 + 行程，返回压缩文本。"""
        if not self.llm_model:
            return ""

        all_history = self.long_term.get_chat_history(limit=max_messages)
        history_from_other_sessions = [
            msg for msg in all_history
            if msg.get("session_id") != self.session_id
        ]

        trip_history = self.long_term.get_trip_history(limit=20)

        if not history_from_other_sessions and not trip_history:
            return ""

        history_text = []
        for msg in history_from_other_sessions[-max_messages:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
            history_text.append(f"[{timestamp}] {role}: {content}")

        history_str = "\n".join(history_text) if history_text else "（无聊天记录）"

        trip_text = []
        for trip in trip_history:
            origin = trip.get("origin", "未知")
            destination = trip.get("destination", "未知")
            start_date = trip.get("start_date", "")
            end_date = trip.get("end_date", "")
            purpose = trip.get("purpose", "旅游")
            timestamp = trip.get("timestamp", "")

            if start_date and end_date:
                trip_text.append(f"[{timestamp}] {origin} → {destination} ({start_date} 至 {end_date}) - {purpose}")
            elif start_date:
                trip_text.append(f"[{timestamp}] {origin} → {destination} ({start_date}) - {purpose}")
            else:
                trip_text.append(f"[{timestamp}] {origin} → {destination} - {purpose}")

        trip_str = "\n".join(trip_text) if trip_text else "（无行程记录）"

        summarization_prompt = f"""请总结以下历史信息中的关键内容，包括：
1. 用户的旅行偏好和习惯
2. 用户询问过的重要问题
3. 用户的出行历史和目的地
4. 其他重要的上下文信息

【历史聊天记录】
{history_str}

【历史行程记录】
{trip_str}

请用简洁的语言总结（不超过200字）："""

        try:
            response = await self.llm_model([{"role": "user", "content": summarization_prompt}])

            summary = ""
            if hasattr(response, '__aiter__'):
                async for chunk in response:
                    if isinstance(chunk, str):
                        summary = chunk
                    elif hasattr(chunk, 'content'):
                        if isinstance(chunk.content, str):
                            summary = chunk.content
                        elif isinstance(chunk.content, list):
                            for item in chunk.content:
                                if isinstance(item, dict) and item.get('type') == 'text':
                                    summary = item.get('text', '')
            elif hasattr(response, 'content'):
                summary = str(response.content)
            else:
                summary = str(response)

            logger.info(f"Generated long-term memory summary ({len(summary)} chars)")
            return summary.strip()

        except Exception as e:
            logger.error(f"Failed to generate long-term summary: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return ""

    def get_long_term_summary(self, max_messages: int = 50) -> str:
        """同步版本的长期记忆摘要（不要在 async 上下文里调用）。"""
        import asyncio

        try:
            asyncio.get_running_loop()
            logger.warning("get_long_term_summary called from async context, please use get_long_term_summary_async instead")
            return ""
        except RuntimeError:
            return asyncio.run(self.get_long_term_summary_async(max_messages))
