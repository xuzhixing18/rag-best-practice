# pip install markdown unstructured
from langchain_community.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader(file_path='../test_doc/GraphRAG和LightRAG详解.md', mode='elements')
data = loader.load()
print(data)

for e in data:
    print("==========")
    print(e.page_content)
