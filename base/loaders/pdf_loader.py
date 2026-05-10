# pip install pdfplumber
from langchain_community.document_loaders import PDFPlumberLoader

# loader = PyPDFLoader(file_path='../test_doc/test.pdf', extract_images=True)
file_path = '../test_doc/test.pdf'
loader = PDFPlumberLoader(file_path=file_path)

# 每一页对应一个document
data = loader.load()
print(data)

for e in data:
    print("=====")
    print(e.page_content)



