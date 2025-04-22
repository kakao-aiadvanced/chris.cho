from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    문서를 청크로 분할합니다.

    Args:
        documents: 분할할 문서 목록

    Returns:
        분할된 문서 청크 목록
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks 