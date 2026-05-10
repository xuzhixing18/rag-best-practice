import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

api_key = os.getenv('GLM_API_KEY')

model = ChatOpenAI(
    model='glm-4-0520',
    temperature='0.6',
    api_key=api_key,
    base_url='https://open.bigmodel.cn/api/paas/v4/'
)

prompt_template = ChatPromptTemplate.from_messages([
    ('system', '你是一个{style}的段子手。用{language}尽你所能回答用户问题。'),
    MessagesPlaceholder(variable_name='user_msg')
])

chain = prompt_template | model

response = chain.invoke({
    'user_msg': [('user', '请给我讲一个程序员的笑话')],
    'style': '幽默风趣',
    'language': '中文'
})
print("===== 完整响应 =====")
print(response)
print("\n===== 回答内容 ====")
print(response.content)