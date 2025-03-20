# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

client = OpenAI(api_key="sk-c2db500c89eb4c42873d583216dd4592", base_url="https://api.deepseek.com/v1")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你的名字叫小农，是一个智能问答助手，可以回答以下相关的问题：农业政策、农业技术、农产品行情、农作物病虫害。"},
        {"role": "user", "content": "你是谁，你能做什么？"},
    ],
    stream=False
)

print(response.choices[0].message.content)