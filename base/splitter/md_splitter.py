from langchain_text_splitters import MarkdownHeaderTextSplitter

with open('../test_doc/test_md_splitter.md',encoding='utf-8') as f:
    text_data = f.read()

label_split = [  # 定义章节的结构
    ('#', '大章节'),
    ('##', '小节'),
    ('###', '章节中的小点'),
    ('####', '小点中的子节点'),
]
# strip_headers: 是否去除标签,为True则去除标签
md_splitter = MarkdownHeaderTextSplitter(label_split, strip_headers=True)

docs = md_splitter.split_text(text_data)

print(f"len: {len(docs)}")
print(docs)

print("***" * 10)

with open('../test_doc/GraphRAG和LightRAG详解.md', encoding='utf-8') as f:
    text_data = f.read()
md_splitter = MarkdownHeaderTextSplitter(label_split, strip_headers=False)
docs = md_splitter.split_text(text_data)

print(f"len: {len(docs)}")
for e in docs:
    print("\n===" * 3)
    print(e)