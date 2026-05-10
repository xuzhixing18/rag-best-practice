from openai import OpenAI
import os
# 使用OpenAI的API

api_key = os.getenv('GLM_API_KEY')
print(api_key)
client = OpenAI(
    api_key=api_key,
    base_url='https://open.bigmodel.cn/api/paas/v4/'
)
response = client.chat.completions.create(
    model="GLM-4.7-Flash",
    messages=[
        {'role': "user", 'content': '请简单介绍下GLM'}
    ],
    temperature=0.1,
    # stream=True
)

print(response)
print(response.choices[0].message.content)