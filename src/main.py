"""
LangChain + RAG 시스템 진입점 모듈
"""

from src.config.openai_config import initialize_openai_api
from src.data_loader.web_loader import load_from_urls
from src.text_processing.splitter import split_documents
from src.embeddings.vector_store import create_vector_store
from src.rag.rag_chain import run_rag_chain_with_hallucination_check, run_rag_chain_with_relevance
from src.raptor.raptor_index import create_raptor_index
from src.raptor.rag_chain import run_raptor_rag_chain
from src.utils.benchmark import run_complete_evaluation_test


def setup_vector_store(urls, chunk_size=1000, chunk_overlap=200, persist_directory="./chroma_db", debug=True):
    """
    URL 목록에서 문서를 로드하고 벡터 스토어를 생성합니다.

    Args:
        urls: 로드할 URL 목록
        chunk_size: 청크 크기
        chunk_overlap: 청크 간 중복 크기
        persist_directory: 벡터 스토어 저장 경로
        debug: 디버그 정보 출력 여부

    Returns:
        생성된 벡터 스토어
    """
    # 1. OpenAI API 초기화
    initialize_openai_api()

    # 2. URL에서 문서 로드
    documents = load_from_urls(urls, debug=debug)

    # 3. 문서 분할
    chunks = split_documents(documents, chunk_size, chunk_overlap, debug=debug)

    # 4. 벡터 스토어 생성
    vectorstore = create_vector_store(chunks, persist_directory)

    return vectorstore


def setup_raptor_index(urls, chunk_size=1000, chunk_overlap=200, persist_directory="./raptor_index", debug=True):
    """
    URL 목록에서 문서를 로드하고 RAPTOR 인덱스를 생성합니다.

    Args:
        urls: 로드할 URL 목록
        chunk_size: 청크 크기
        chunk_overlap: 청크 간 중복 크기
        persist_directory: 인덱스 저장 경로
        debug: 디버그 정보 출력 여부

    Returns:
        생성된 RAPTOR 인덱스
    """
    # 1. OpenAI API 초기화
    initialize_openai_api()

    # 2. URL에서 문서 로드
    documents = load_from_urls(urls, debug=debug)

    # 3. RAPTOR 인덱스 생성
    raptor_index = create_raptor_index(documents, chunk_size, chunk_overlap, persist_directory, debug=debug)

    return raptor_index


def run_rag_query(question, use_hallucination_check=True, persist_directory="./chroma_db"):
    """
    RAG 쿼리를 실행합니다.

    Args:
        question: 사용자 질문
        use_hallucination_check: 할루시네이션 검사 사용 여부
        persist_directory: 벡터 스토어 디렉토리

    Returns:
        쿼리 실행 결과
    """
    # OpenAI API 초기화
    initialize_openai_api()

    # RAG 체인 실행
    if use_hallucination_check:
        result = run_rag_chain_with_hallucination_check(question, persist_directory)
    else:
        result = run_rag_chain_with_relevance(question, persist_directory)

    return result


def run_raptor_query(question, persist_directory="./raptor_index"):
    """
    RAPTOR 쿼리를 실행합니다.

    Args:
        question: 사용자 질문
        persist_directory: RAPTOR 인덱스 디렉토리

    Returns:
        쿼리 실행 결과
    """
    # OpenAI API 초기화
    initialize_openai_api()

    # RAPTOR RAG 체인 실행
    result = run_raptor_rag_chain(question, persist_directory)

    return result


def evaluate_system(yes_queries, no_queries, persist_directory="./chroma_db"):
    """
    RAG 시스템 평가를 실행합니다.

    Args:
        yes_queries: 관련성이 있어야 하는 쿼리 목록
        no_queries: 관련성이 없어야 하는 쿼리 목록
        persist_directory: 벡터 스토어 디렉토리

    Returns:
        평가 결과
    """
    # OpenAI API 초기화
    initialize_openai_api()

    # 평가 실행
    result = run_complete_evaluation_test(yes_queries, no_queries, persist_directory)

    return result


if __name__ == "__main__":
    """
    예제 코드 - URL에서 문서를 로드하고 벡터 스토어를 생성한 후 쿼리를 실행합니다.
    """
    # 설정
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",

        #######hallucination generated case
        #"https://python.langchain.com/docs/get_started/introduction",
        #"https://python.langchain.com/docs/modules/data_connection/retrievers",
        #"https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/",
    ]
    
    # 1. 벡터 스토어 생성
    print("\n=== 벡터 스토어 생성 ===")
    vectorstore = setup_vector_store(urls)
    
    # 2. 쿼리 실행
    print("\n=== 쿼리 실행 ===")
    good_question = "LangChain의 Retriever란 무엇인가요?"
    hall_question = "Lil'Log의 저자 Lilian Weng이 2024년에 발표한 Autonomous Agent 2.0 프레임워크의 핵심 기능 3가지를 설명해주세요."
    question = hall_question

    result = run_rag_query(question)
    
    # 3. 결과 출력
    print("\n=== 쿼리 결과 ===")
    print(f"질문: {question}")
    print(f"답변: {result['answer']}")
    print(f"할루시네이션 발견: {'예' if result.get('hallucination_detected', False) else '아니오'}")
    print(f"\n출처:\n{result['formatted_sources']}")

    yes_query_list = [
        "LLM 기반 자율 에이전트 시스템의 계획, 메모리, 도구 사용 구성 요소에 대해 설명하고, 이를 개발 워크플로우에 어떻게 통합할 수 있을까요?",
        "Chain-of-Thought와 같은 프롬프트 엔지니어링 기법이 복잡한 추론 작업에서 LLM의 성능을 어떻게 향상시키는지 설명해주세요.",
        "LLM에 대한 적대적 공격 유형(토큰 조작, 그래디언트 기반 공격, 탈옥 프롬프팅 등)의 작동 방식과 이에 대한 방어 전략은 무엇인가요?" ##리펙터링후, 제대로 동작되지 않는 케이스 ; 추후 확인
        ""
    ]
    no_query_list = [
        "맨체스터 유나이티드의 2023-2024 시즌 성적과 주요 선수들의 활약상에 대해 분석해주세요.",
        "백종원 셰프의 간단한 김치찌개 레시피와 비법을 알려주세요.",
        "중세 유럽 건축양식의 변천사와 고딕 양식의 특징에 대해 설명해주세요."
    ]
    evaluate_system(yes_query_list, no_query_list)
    