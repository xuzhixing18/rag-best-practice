import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from utils.env_utils import OPENAI_API_KEY

openai_embedding = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base="https://xiaoai.plus/v1"
)

bge_embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

def cosine_similarity(vec1, vec2):
    """计算两个向量之间的余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


if __name__ == '__main__':
    similar_pairs = [
        ("人工智能正在改变世界", "AI技术正在重塑全球格局"),
        ("机器学习是人工智能的一个分支", "ML属于AI的子领域"),
        ("深度学习需要大量数据", "神经网络训练依赖大数据集"),
        ("Python是流行的编程语言", "Python被广泛用于软件开发")
    ]

    print("=" * 10)
    print("测试1：语义相似但表述不同的句子")
    print("=" * 10)
    for text1, text2 in similar_pairs:
        emb1 = bge_embedding.embed_query(text1)
        emb2 = bge_embedding.embed_query(text2)
        similarity = cosine_similarity(emb1, emb2)
        print(f"文本1: {text1}")
        print(f"文本2: {text2}")
        print(f"相似度: {similarity:.4f}")
        print("*" * 20)

    # 测试用例2：语义不相关的句子
    print("\n" + "=" * 10)
    print("测试2：语义不相关的句子（低相似度）")
    print("=" * 10)

    dissimilar_pairs = [
        ("人工智能正在改变世界", "今天天气真好"),
        ("机器学习需要数据", "我喜欢吃苹果"),
        ("Python编程语言", "篮球比赛很精彩"),
        ("深度学习算法", "音乐使人愉悦")
    ]

    for text1, text2 in dissimilar_pairs:
        emb1 = bge_embedding.embed_query(text1)
        emb2 = bge_embedding.embed_query(text2)
        similarity = cosine_similarity(emb1, emb2)
        print(f"文本1: {text1}")
        print(f"文本2: {text2}")
        print(f"相似度: {similarity:.4f}")
        print("*" * 20)


    # 测试用例3：多文档嵌入和检索场景
    print("\n" + "=" * 60)
    print("测试3：模拟RAG检索场景")
    print("=" * 60)

    documents = [
        "人工智能(AI)是计算机科学的一个分支，旨在创造智能机器",
        "机器学习是一种让计算机从数据中学习的技术",
        "深度学习是机器学习的一种，使用多层神经网络",
        "自然语言处理使计算机能够理解和生成人类语言",
        "计算机视觉让机器能够'看'和理解图像内容",
        "推荐系统根据用户行为预测用户偏好",
        "自动驾驶汽车使用传感器和算法来导航道路",
        "医疗AI可以帮助医生诊断疾病和制定治疗方案"
    ]

    query = "如何让计算机理解人类语言？"

    # 计算查询与所有文档的相似度
    query_embedding = bge_embedding.embed_query(query)
    doc_embeddings = bge_embedding.embed_documents(documents)

    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        sim = cosine_similarity(query_embedding, doc_emb)
        similarities.append((i, sim, documents[i]))

    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"查询: {query}\n")
    print("最相关的文档（按相似度排序）:")
    for rank, (idx, sim, doc) in enumerate(similarities[:3], 1):
        print(f"{rank}. 相似度: {sim:.4f}")
        print(f"   文档: {doc}")
        print()