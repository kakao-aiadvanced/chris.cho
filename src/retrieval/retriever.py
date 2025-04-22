"""
벡터 스토어로부터 관련 문서를 검색하는 기능을 제공하는 모듈
"""

from typing import List
from langchain_chroma import Chroma
from src.embeddings.vector_store import get_embedding_model

def get_retriever(persist_directory: str = "./chroma_db", k: int = 6) -> callable:
    """
    벡터 스토어로부터 retriever를 생성합니다.

    Args:
        persist_directory: 벡터 스토어 디렉토리
        k: 검색할 문서 수

    Returns:
        생성된 retriever
    """
    # 임베딩 모델 초기화
    embedding = get_embedding_model()

    # 기존 벡터 스토어 로드
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )

    # Retriever 생성
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    return retriever


def format_docs(docs: List) -> str:
    """
    문서 목록을 텍스트로 포맷팅합니다.

    Args:
        docs: 포맷팅할 문서 목록

    Returns:
        포맷팅된 텍스트
    """
    return "\n\n".join(doc.page_content for doc in docs)


def format_sources(sources: List) -> str:
    """
    출처 목록을 텍스트로 포맷팅합니다.

    Args:
        sources: 포맷팅할 출처 목록

    Returns:
        포맷팅된 출처 텍스트
    """
    if not sources:
        return "출처 없음"

    formatted_sources = []
    for i, source in enumerate(sources):
        if hasattr(source, 'metadata') and 'source' in source.metadata:
            source_url = source.metadata['source']
            formatted_sources.append(f"{i+1}. {source_url}")

    if not formatted_sources:
        return "출처 정보 없음"

    return "\n".join(formatted_sources) 