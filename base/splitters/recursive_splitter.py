from langchain_text_splitters import RecursiveCharacterTextSplitter

with open('../test_doc/GraphRAG和LightRAG详解.md',encoding='utf-8') as f:
    text_data = f.read()

# 递归切割器
# ["\n\n", "\n", " ", ""]
# chunk_size：分块最大的大小，其大小由length_function决定
# chunk_overlap：数据块之间的目标重叠。重叠数据块有助于在数据块之间划分上下文时减少信息丢失
#          chunk_overlap值一般设置为chunk_size的10%~15%
# length_function：确定块大小的函数,如果要按照单词/中文,可结合jieba等分词器来自定义切割逻辑
# is_separator_regex：分隔符列表（默认为 ）是否应解释为正则表达式
# separators: 递归分块的规则,不指定,则默认使用["\n\n", "\n", " ", ""]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75,
    # 使用len函数按照字符切分
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.create_documents([text_data])

print(len(chunks))

for e in chunks[:6]:
    print("===" * 5)
    print(e)

print("~~~~~~~~~~~~~\n\n")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75,
    length_function=len,
    is_separator_regex=False,
    separators=["#","##","###", "\n\n","\n", "。",".", "，",","]
)

chunks = splitter.create_documents([text_data])

for e in chunks[:6]:
    print("===" * 10)
    print(e)