import json
import time

from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage

llm = OpenAILike(
    model="deepseek-chat",
    api_base="https://api.deepseek.com/v1",
    api_key="sk-c2db500c89eb4c42873d583216dd4592",
    temperature=0.3,
    max_tokens=1024,
    timeout=60,
    is_chat_model=True,  # 适用于对话模型
    additional_kwargs={"stop": ["<|im_end|>"]}  # DeepSeek的特殊停止符
)

messages = [
    ChatMessage(
        role="system", content="你的名字叫小农，是一个智能问答助手，可以回答以下相关的问题：农业政策、农业技术、农产品行情、农作物病虫害。"
    ),
    ChatMessage(
        role="user", content="水稻常见病虫害有哪些"
    ),
]

rsp = llm.chat(messages)
result = rsp.message.content
print(type(result), result)
response = {"success": True,
            "message": "success",
            "code": 200,
            "timestamp": int(time.time()),
            "result": result}
json.dumps(response)
