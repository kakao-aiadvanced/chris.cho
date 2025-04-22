from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def create_vector_store(chunks: List[Dict[str, Any]], persist_directory: str = "./chroma_db") -> Chroma:
    """
    문서 청크로부터 벡터 스토어를 생성합니다.

    Args:
        chunks: 문서 청크 목록
        persist_directory: 벡터 스토어를 저장할 디렉토리

    Returns:
        생성된 벡터 스토어
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    return vectorstore 