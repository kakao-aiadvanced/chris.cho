"""
청크로부터 벡터 스토어를 생성하고 관리하는 모듈
"""

from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.config.openai_config import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_DIMENSIONS

def create_vector_store(chunks: List, persist_directory: str = "./chroma_db", model: str = DEFAULT_EMBEDDING_MODEL,
                       dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS) -> Chroma:
    """
    청크로부터 벡터 스토어를 생성합니다.

    Args:
        chunks: 벡터화할 청크 목록
        persist_directory: 벡터 스토어 저장 경로
        model: 임베딩 모델명
        dimensions: 임베딩 차원 수

    Returns:
        생성된 벡터 스토어
    """
    embedding = OpenAIEmbeddings(
        model=model,
        dimensions=dimensions
    )

    print(f"\n임베딩 모델 초기화 완료: {model}")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_directory
    )

    print(f"벡터 스토어 생성 완료: {len(chunks)}개 청크가 임베딩되어 저장됨")
    print(f"저장 위치: {persist_directory}")

    return vectorstore


def get_embedding_model(model: str = DEFAULT_EMBEDDING_MODEL, dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS) -> OpenAIEmbeddings:
    """
    OpenAI 임베딩 모델을 생성합니다.

    Args:
        model: 임베딩 모델명
        dimensions: 임베딩 차원 수

    Returns:
        OpenAI 임베딩 모델 인스턴스
    """
    return OpenAIEmbeddings(
        model=model,
        dimensions=dimensions
    ) 