import os

from zhipuai import ZhipuAI

api_key = os.getenv("GLM_API_KEY")

print(api_key)

client = ZhipuAI(api_key=api_key)

response = client.chat.completions.create(
    model="GLM-4.7-Flash",
    messages=[
        {"role": "user", "content": "请简单介绍RAG"},
    ],
    # temperature 控制输出的随机性,
    # 取值范围0-1之间，值越小越稳定,越一致;值越大,越随机,越发散
    temperature=0.4,
    stream=True,
)

# print(response)
# print("==========")
# print(response.choices[0].message.content)

# 开启Stream流式输出
for s in response:
    print(s.choices[0].delta.content)