"""
웹 URL로부터 데이터를 로드하는 모듈
"""

import time
from typing import List, Dict, Any
from langchain_community.document_loaders import WebBaseLoader

def load_from_urls(urls: List[str], debug: bool = False) -> List:
    """
    URL 목록에서 문서를 로드합니다.

    Args:
        urls: 로드할 URL 목록
        debug: 디버그 정보 출력 여부

    Returns:
        로드된 문서 목록
    """
    print("=== 순차적 로딩 ===")
    all_docs = []

    for url in urls:
        print(f"\n로딩 중: {url}")
        loader = WebBaseLoader(url)

        try:
            docs = loader.load()
            all_docs.extend(docs)

            if debug:
                for doc in docs:
                    print(f"- 메타데이터: {doc.metadata}")
                    print(f"- 내용 미리보기: {doc.page_content[:128]}...\n")

            # 선택적: 서버 부하 방지를 위한 지연
            # time.sleep(1)

        except Exception as e:
            print(f"오류 발생 ({url}): {e}")

    print(f"\n총 로드된 문서 수: {len(all_docs)}")
    return all_docs 