from typing import List, Dict, Any
from langchain_community.document_loaders import WebBaseLoader

def load(urls: List[str]) -> List[Dict[str, Any]]:
    """
    주어진 URL 목록에서 문서를 로드합니다.

    Args:
        urls: 로드할 URL 목록

    Returns:
        로드된 문서 목록
    """
    loader = WebBaseLoader(urls)
    documents = loader.load()
    return documents 