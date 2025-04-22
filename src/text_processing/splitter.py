"""
문서를 청크로 분할하는 기능을 제공하는 모듈
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents: List, chunk_size: int = 1000, chunk_overlap: int = 200, debug: bool = False) -> List:
    """
    문서를 청크로 분할합니다.

    Args:
        documents: 분할할 문서 목록
        chunk_size: 청크 크기
        chunk_overlap: 청크 간 중복 크기
        debug: 디버그 정보 출력 여부

    Returns:
        분할된 청크 목록
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)

    if debug:
        print(f"원본 문서 수: {len(documents)}")
        print(f"분할 후 청크 수: {len(chunks)}")

        print("\n== 처음 3개 청크 개요 ==")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n청크 #{i + 1}")
            print(f"소스: {chunk.metadata.get('source', '출처 없음')}")
            print(f"문자 수: {len(chunk.page_content)}")
            print(f"내용 미리보기: {chunk.page_content[:100]}...")

    return chunks


def group_chunks_by_source(chunks: List) -> dict:
    """
    청크를 소스별로 그룹화합니다.

    Args:
        chunks: 그룹화할 청크 목록

    Returns:
        소스별로 그룹화된 청크 사전
    """
    groups = {}
    for chunk in chunks:
        source = chunk.metadata.get('source', 'unknown')
        if source not in groups:
            groups[source] = []
        groups[source].append(chunk)
    return groups 