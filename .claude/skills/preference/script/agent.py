"""
偏好智能体
职责：收集用户的长期偏好
如"我的家在XXX"、"我喜欢XXX酒店"
"""
import json
import logging
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

logger = logging.getLogger(__name__)


class PreferenceAgent:
    """偏好智能体"""

    def __init__(self, name: str = "PreferenceAgent", model=None, memory_manager=None, **kwargs):
        self.name = name
        self.model = model
        self.memory_manager = memory_manager
        from utils.skill_loader import SkillLoader
        self.skill_loader = SkillLoader()

    async def run(self, input_data: dict) -> dict:
        data = input_data
        context = data.get("context", {})
        user_query = context.get("rewritten_query", "") or str(data)

        # 获取当前已保存的偏好
        current_preferences = {}
        if self.memory_manager:
            current_preferences = self.memory_manager.long_term.get_preference()

        # 格式化当前偏好，便于展示
        current_prefs_str = json.dumps(current_preferences, ensure_ascii=False, indent=2)

        # 动态读取 Prompt 指令 (Progressive Disclosure)
        skill_instruction = self.skill_loader.get_skill_content("preference")
        if not skill_instruction:
            skill_instruction = "请分析用户的偏好。"

        prompt = f"""你是用户偏好分析专家，负责提取用户的长期偏好信息。

【当前已保存的用户偏好】
{current_prefs_str}

【新的用户输入】
{user_query}

【任务说明】
{skill_instruction}

请直接输出JSON：
"""

        try:
            # 调用模型
            response = await self.model.ainvoke([
                {"role": "user", "content": prompt}
            ])
            text = response.content

            # 清理文本，移除markdown代码块标记
            text = text.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()

            # 提取JSON
            start_idx = text.find('{')
            end_idx = text.rfind('}')

            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx:end_idx+1]
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError as e:
                    # 记录详细错误信息用于调试
                    logger.error(f"JSON parse failed. Text sample: {json_str[:100]}")
                    raise ValueError(f"Failed to parse JSON. Error: {e}")
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            logger.error(f"Preference collection failed: {e}")
            result = {"has_preferences": False, "error": str(e)}

        return result
