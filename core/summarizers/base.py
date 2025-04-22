import requests
from typing import List, Optional
import logging
from config import settings
from openai import OpenAI
logger = logging.getLogger(__name__)
import json
class BaseSummarizer:
    """所有摘要策略的基类"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key or settings.API_KEY
        self.API_URL = settings.API_URL
        self.MODEL = settings.MODEL
    
    def summarize(self, documents: List[str]) -> str:
        """需要子类实现的具体策略"""
        raise NotImplementedError
        
    def _generate_prompt(self, documents_text: str) -> str:
        """使用通用模板生成prompt"""
        return settings.SUMMARY_PROMPT_TEMPLATE.format(
            documents_text=documents_text
        )
        
    # def _call_api(self, prompt: str) -> Optional[str]:
    #     """与DeepSeek API交互的公共方法"""
    #     headers = {
    #         "Authorization": f"Bearer {self.api_key}",
    #         "Content-Type": "application/json"
    #     }
    #
    #     payload = {
    #         "model": self.MODEL,
    #         "messages": [
    #             {"role": "system", "content": settings.SYSTEM_PROMPT_TEMPLATE},
    #             {"role": "user", "content": prompt}
    #         ],
    #         "temperature": settings.TEMPERATURE,
    #         "max_tokens": settings.MAX_TOKENS
    #     }
    #
    #     try:
    #         response = requests.post(self.API_URL, json=payload, headers=headers)
    #         response.raise_for_status()
    #         return response.json()['choices'][0]['message']['content'].strip()
    #     except Exception as e:
    #         logger.error(f"DeepSeek API request failed: {str(e)}")
    #         return None

    def _call_api(self, prompt: str) -> Optional[str]:
        """与Qwen API交互的公共方法"""
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.API_URL
        )
        messages = [
                {"role": "system", "content": settings.SYSTEM_PROMPT_TEMPLATE},
                {"role": "user", "content": prompt}
            ]
        try:
            response = client.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
                #stream=True# 若选择qwq32b,则设为True，因为qwq 模型仅支持流式输出方式调用
            )
            # answer_content = ""# 因为是流式的，所以定义完整回复
            # for chunk in response:
            #     content = chunk.choices[0].delta.content  # 获取回答
            #     if (content != None):  # 如果回复不为空，则加入到完整回复中。因为qwq先回答思维链，再回答问题，所以需要判断
            #         answer_content += chunk.choices[0].delta.content
            # return answer_content

            content_josn = json.loads(response.model_dump_json())
            return content_josn["choices"][0]["message"]["content"].strip()

        except Exception as e:
            logger.error(f"Qwen API request failed: {str(e)}")
            return None