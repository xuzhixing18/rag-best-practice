from langchain_community.embeddings import  BaichuanTextEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# 创建嵌入模型
embeddings = BaichuanTextEmbeddings()

# SemanticChunker关键参数:
#  1) breakpoint_threshold_type参数: 用于定义如何确定切割点的阈值计算方法
#  可选值:
#   'percentile'：基于所有相似度分数的百分位数
#   'standard_deviation'：基于标准差
#   'interquartile'：基于四分位距
#   'gradient'：基于梯度变化
#  2) breakpoint_threshold_amount 参数，用于设置具体的阈值数值
#     当相邻句子的相似度低于75%的分位数时，在该位置进行切割
#  调整建议:
#   值越大：切割越保守，分块越少但可能更大
#   值越小：切割越激进，分块越多但可能更小
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount = 75.0
)

with open('../test_doc/文档切割策略详解.md', encoding='utf-8') as f:
    text_data = f.read()

docs = text_splitter.create_documents([text_data])

print(f"len: {len(docs)}")
i = 0
for e in docs:
    i += 1
    print(f"**** doc {i} ****")
    print(e)