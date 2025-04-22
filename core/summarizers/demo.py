import json
from typing import Optional

from openai import OpenAI
QWEN_API_KEY = "sk-5bcedca0a08c4efda08042c674199b60"
QWEN_MODEL = "qwen-plus"
QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def call_api(prompt: str) -> Optional[str]:
    """与Qwen API交互的公共方法"""
    client = OpenAI(
        api_key=QWEN_API_KEY,
        base_url=QWEN_API_URL
    )
    messages =  [
            {"role": "system", "content": "你是一只米老鼠"},
            {"role": "user", "content": prompt}
        ]

    try:
        response = client.chat.completions.create(
            model=QWEN_MODEL,  # 此处以qwq-32b为例，可按需更换模型名称
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
            #stream=True,# QwQ 模型仅支持流式输出方式调用
        )
        # answer_content = ""
        # for chunk in response:
        #     content = chunk.choices[0].delta.content#获取回答
        #     if (content != None):# 如果回复不为空，则加入到完整回复中
        #         answer_content += chunk.choices[0].delta.content
        # return answer_content
        content_josn=json.loads(response.model_dump_json())
        return content_josn["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(e)
        return None

answer=call_api("你好")
print(answer)

