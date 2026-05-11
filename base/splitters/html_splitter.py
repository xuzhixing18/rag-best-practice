from langchain_text_splitters import HTMLHeaderTextSplitter

with open('../test_doc/test_splitter.html',encoding='utf-8') as f:
    html_doc = f.read()

# 定义章节的结构
label_split = [
    ('h1', '大章节 1'),
    ('h2', '小节 2'),
    ('h3', '章节中的小点 3'),
]

html_splitter = HTMLHeaderTextSplitter(label_split)

chunks = html_splitter.split_text(html_doc)

for e in chunks:
    print("===" * 3)
    print(e)

print("***" * 5)

label_split2 = [  # 定义章节的结构
    ('h1', '大章节'),
    ('h2', '小节'),
    ('h3', '章节中的小点'),
    ('h4', '小点中的子节点'),
]
html_splitter = HTMLHeaderTextSplitter(label_split2)

docs = html_splitter.split_text_from_url('https://cn.vuejs.org/guide/introduction.html')

print('--------------------')
print('总共有多少个docs： ', len(docs))
print('--------------------------------')

for e in docs:
    print("===" * 5)
    print(e)
