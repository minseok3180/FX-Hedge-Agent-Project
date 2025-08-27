import os
from typing import List
import pandas as pd


def build_rag_db(vector_db_dir: str, embedding_model: str, ts_csv: str, news_csv: str) -> str:
    """
    정형(시계열 특징 요약) + 비정형(뉴스)을 문서화하여 Chroma에 저장.
    """
    os.makedirs(vector_db_dir, exist_ok=True)
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.docstore.document import Document

    embed = HuggingFaceEmbeddings(model_name=embedding_model)
    docs: List[Document] = []

    # 시계열 요약
    ts = pd.read_csv(ts_csv)
    stats = ts["ret"].describe().to_string()
    docs.append(Document(page_content=f"[타입:timeseries_summary]\n기술통계:\n{stats}", metadata={"type": "ts_summary"}))

    # 뉴스
    news = pd.read_csv(news_csv)
    for _, row in news.iterrows():
        docs.append(
            Document(
                page_content=f"[타입:news]\n제목:{row['title']}\n본문:{row['content']}",
                metadata={"type": "news", "date": str(row["date"])},
            )
        )

    Chroma.from_documents(documents=docs, embedding=embed, persist_directory=vector_db_dir)
    return vector_db_dir


def get_retriever(vector_db_dir: str, embedding_model: str):
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embed = HuggingFaceEmbeddings(model_name=embedding_model)
    vs = Chroma(persist_directory=vector_db_dir, embedding_function=embed)
    return vs.as_retriever(search_kwargs={"k": 5})


