from config.open_api_config import initialize_openai_api
initialize_openai_api()

from typing import Dict, Any, List, Tuple
import bs4
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
import time

def load(urls, debug=False):

    # 1. 순차적으로 각 URL 로딩
    print("=== 순차적 로딩 ===")
    all_docs = []

    for url in urls:
        print(f"\n로딩 중: {url}")

        # 단일 URL에 대한 WebBaseLoader 인스턴스 생성
        loader = WebBaseLoader(url)

        # SSL 인증 오류 방지 옵션 (필요시 주석 해제)
        # loader.requests_kwargs = {'verify': False}

        # URL에서 문서 로드
        try:
            docs = loader.load()
            all_docs.extend(docs)

            # 각 문서의 메타데이터와 내용 일부 출력
            if debug:
                for doc in docs:
                    print(f"- 메타데이터: {doc.metadata}")
                    print(f"- 내용 미리보기: {doc.page_content[:128]}...\n")

            # 서버에 부담을 주지 않기 위해 요청 간 간격 추가
            #time.sleep(1)

        except Exception as e:
            print(f"오류 발생 ({url}): {e}")

    print(f"\n총 로드된 문서 수: {len(all_docs)}")
    return all_docs

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    RecursiveCharacterTextSplitter를 사용하여 문서를 분할합니다.

    Args:
        documents: 분할할 문서 리스트
        chunk_size: 각 청크의 목표 크기 (기본값: 1000)
        chunk_overlap: 연속된 청크 간의 중복 크기 (기본값: 200)

    Returns:
        분할된 청크 리스트
    """
    # 분할자(splitter) 초기화
    # RecursiveCharacterTextSplitter는 여러 구분자(separators)를 우선순위에 따라 사용
    text_splitter = RecursiveCharacterTextSplitter(
        # 구분자와 우선순위:
        # 먼저 "\n\n"로 구분, 다음 "\n", 다음 " ", 마지막으로 ""(단일 문자)
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,  # 각 청크의 목표 크기
        chunk_overlap=chunk_overlap,  # 연속된 청크 간의 중복
        length_function=len,  # 텍스트 길이 측정 함수
        is_separator_regex=False,  # 구분자를 정규식으로 취급하지 않음
    )

    # 문서 분할 실행
    chunks = text_splitter.split_documents(documents)

    print(f"원본 문서 수: {len(documents)}")
    print(f"분할 후 청크 수: {len(chunks)}")

    # 청크 개요 출력 (처음 3개)
    print("\n== 처음 3개 청크 개요 ==")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n청크 #{i + 1}")
        print(f"소스: {chunk.metadata.get('source', '출처 없음')}")
        print(f"문자 수: {len(chunk.page_content)}")
        print(f"내용 미리보기: {chunk.page_content[:100]}...")

    return chunks


# 3. 임베딩 및 벡터 스토어 생성 함수
def create_vector_store(chunks):
    """
    OpenAI 임베딩을 사용하여 청크를 임베딩하고 Chroma 벡터 스토어에 저장합니다.

    Args:
        chunks: 문서 청크 리스트

    Returns:
        생성된 Chroma 벡터 스토어
    """
    # OpenAI 임베딩 모델 초기화 (text-embedding-3-small 사용)
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",  # 요구사항에 맞는 모델 사용
        dimensions=1536  # text-embedding-3-small의 기본 차원
    )

    print("\n임베딩 모델 초기화 완료: text-embedding-3-small")

    # Chroma 벡터 스토어 생성
    # persist_directory 설정 시 디스크에 영구 저장됨 (선택적)
    persist_directory = "./chroma_db"

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_directory
    )

    # 최신 버전의 Chroma에서는 persist() 메서드를 호출할 필요가 없습니다
    # persist_directory를 지정하면 자동으로 저장됩니다

    print(f"벡터 스토어 생성 완료: {len(chunks)}개 청크가 임베딩되어 저장됨")
    print(f"저장 위치: {persist_directory}")

    return vectorstore


def retrieve_relevant_chunks(query, persist_directory="./chroma_db", debug=False):
    """
    사용자 쿼리를 받아 벡터 스토어에서 관련 청크를 검색합니다.

    Args:
        query: 검색 쿼리 문자열
        persist_directory: Chroma 벡터 스토어가 저장된 디렉토리 경로

    Returns:
        검색된 관련 문서 목록
    """
    print(f"쿼리: '{query}'로 관련 청크 검색 중...")

    # OpenAI 임베딩 모델 초기화 (text-embedding-3-small 사용)
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536
    )

    # 기존 벡터 스토어 로드
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )

    # Retriever 생성 - 요구사항대로 similarity 검색, k=6 설정
    # 유사도 점수 관련 파라미터 제거
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # 유사도 기반 검색
        search_kwargs={"k": 6}  # 상위 6개 결과 반환
    )

    # 쿼리 실행하여 관련 문서 검색
    relevant_docs = retriever.invoke(query)

    # 유사도 점수 직접 계산 시도
    try:
        # 임베딩 생성
        query_embedding = embedding.embed_query(query)

        # 각 청크의 유사도를 계산하기 위한 함수
        def compute_similarity(doc_text):
            doc_embedding = embedding.embed_query(doc_text)
            # 코사인 유사도 계산 (간단한 구현)
            import numpy as np
            return np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )

        # 각 문서에 유사도 점수 추가
        for doc in relevant_docs:
            try:
                # 유사도 계산 및 메타데이터에 저장
                similarity = compute_similarity(doc.page_content)
                doc.metadata["similarity_score"] = float(similarity)
            except Exception as e:
                print(f"문서 유사도 계산 중 오류: {e}")
                doc.metadata["similarity_score"] = "계산 실패"

    except Exception as e:
        print(f"유사도 계산 시도 중 오류: {e}")
        # 오류 발생 시 진행
        pass

    # 검색 결과 정보 출력
    print(f"\n검색 결과: {len(relevant_docs)}개의 관련 청크 발견\n")

    # 각 검색 결과 출력
    if debug:
        for i, doc in enumerate(relevant_docs):
            print(f"=== 관련 청크 #{i + 1} ===")
            print(f"출처: {doc.metadata.get('source', '출처 없음')}")
            print(f"유사도 점수: {doc.metadata.get('similarity_score', '점수 정보 없음')}")
            # 내용이 너무 길면 일부만 표시
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"내용 미리보기: {content_preview}")
            print("-" * 50)

    return relevant_docs


def create_relevance_evaluation_chain():
    """
    사용자 쿼리와 검색된 청크 간의 관련성을 평가하기 위한 체인을 생성합니다.
    """
    # LLM 모델 초기화
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    # 간단한 JSON 스키마와 커스텀 파서 생성
    parser = JsonOutputParser()

    # 프롬프트 템플릿 생성 - 모든 중괄호 이스케이프 처리
    relevance_template = """<im_start>system
You are an expert in evaluating information retrieval quality for AI and language model topics. Your task is to assess whether the retrieved context is relevant to the user's query about AI technologies.

Guidelines:
1. For AI and language model related queries, take a broad view of relevance.
2. ANY context that discusses language models (LLMs), AI capabilities, or machine learning techniques should be considered relevant ('yes') to queries about similar AI technologies.
3. Specifically, if the query mentions prompting techniques (like Chain-of-Thought) and the context mentions ANY LLM application, system architecture, or implementation (like HuggingGPT), consider them relevant ('yes') as they are part of the same AI ecosystem.
4. Both prompt engineering techniques and LLM application frameworks are considered closely related topics within AI research.
5. Only mark as not relevant ('no') if the context discusses a completely non-AI domain (like cooking recipes, sports, or history) with no connection to machine learning or AI systems.
6. Consider the educational value - if someone interested in the query topic would benefit from knowing the information in the context, mark it relevant ('yes').
7. Your output must be valid JSON with exactly one key named "relevance" and value either "yes" or "no".

Always respond with this exact JSON format:
{{"relevance": "yes"}} if relevant, or {{"relevance": "no"}} if not relevant.
<im_end>

<im_start>user
User Query: {question}

Retrieved Context: 
{context}
<im_end>

<im_start>assistant
"""

    # 프롬프트 템플릿 생성 - format_instructions 제거
    relevance_prompt = PromptTemplate(
        template=relevance_template,
        input_variables=["question", "context"],
    )

    # 관련성 평가 체인 생성
    relevance_chain = relevance_prompt | llm | parser

    return relevance_chain


def retrieve_and_evaluate_relevance(query: str, persist_directory: str = "./chroma_db", debug=False) -> List[Dict[str, Any]]:
    """
    사용자 쿼리에 대해 벡터 스토어에서 관련 청크를 검색하고 관련성을 평가합니다.

    Args:
        query: 사용자 쿼리 문자열
        persist_directory: Chroma 벡터 스토어가 저장된 경로

    Returns:
        검색 및 평가 결과 리스트
    """
    # 임베딩 모델 초기화
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536
    )

    # 벡터 스토어 로드
    try:
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )
        print(f"벡터 스토어를 로드했습니다: {persist_directory}")
    except Exception as e:
        print(f"벡터 스토어 로드 중 오류 발생: {e}")
        return []

    # Retriever 생성
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    # 쿼리 실행
    try:
        retrieved_chunks = retriever.invoke(query)
        print(f"{len(retrieved_chunks)}개의 청크를 검색했습니다.")
    except Exception as e:
        print(f"검색 중 오류 발생: {e}")
        return []

    # 관련성 평가 체인 생성
    relevance_chain = create_relevance_evaluation_chain()

    # 각 청크의 관련성 평가
    results = []
    for i, chunk in enumerate(retrieved_chunks):
        print(f"\n청크 #{i + 1} 평가 중...")
        try:
            if debug:
                print(f"=== 관련 청크 #{i + 1} ===")
                print(f"출처: {chunk.metadata.get('source', '출처 없음')}")
                print(f"유사도 점수: {chunk.metadata.get('similarity_score', '점수 없음')}")  # 유사도 점수 출력
                # 내용이 너무 길면 일부만 표시
                content_preview = chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content
                print(f"내용 미리보기: {content_preview}")
                print("-" * 50)


            # 관련성 평가 실행
            evaluation = relevance_chain.invoke({
                "question": query,
                "context": chunk.page_content
            })

            # 결과 저장
            results.append({
                "chunk_index": i,
                "source": chunk.metadata.get("source", "출처 없음"),
                "content_preview": chunk.page_content[:150] + ("..." if len(chunk.page_content) > 150 else ""),
                "evaluation": evaluation
            })

            # 결과 출력
            print(f"  출처: {chunk.metadata.get('source', '출처 없음')}")
            print(f"  평가 결과: {evaluation}")

        except Exception as e:
            print(f"  평가 중 오류 발생: {e}")
            results.append({
                "chunk_index": i,
                "source": chunk.metadata.get("source", "출처 없음"),
                "content_preview": chunk.page_content[:150] + ("..." if len(chunk.page_content) > 150 else ""),
                "evaluation": {"relevance": "error", "error": str(e)}
            })

    # 관련성 통계 계산
    relevant_count = sum(1 for r in results if r["evaluation"].get("relevance") == "yes")
    non_relevant_count = sum(1 for r in results if r["evaluation"].get("relevance") == "no")
    error_count = len(results) - relevant_count - non_relevant_count

    # 결과 요약 출력
    print("\n=== 평가 결과 요약 ===")
    print(f"총 검색된 청크: {len(results)}")
    print(f"관련성 있음: {relevant_count}")
    print(f"관련성 없음: {non_relevant_count}")
    print(f"평가 중 오류: {error_count}")

    return results


def test_queries_for_expected_relevance(query_list: List[str], expected_result: str,
                                        persist_directory: str = "./chroma_db") -> Tuple[int, int, List[Dict]]:
    """
    쿼리 목록에 대한 평가를 실행하고 기대 결과와 일치하는지 확인합니다.

    Args:
        query_list: 테스트할 쿼리 목록
        expected_result: 기대되는 결과 ('yes' 또는 'no')
        persist_directory: 벡터 스토어 디렉토리

    Returns:
        (일치 수, 총 평가 수, 상세 결과 목록)
    """
    total_evaluations = 0
    matching_evaluations = 0
    detailed_results = []

    # 각 쿼리 실행
    for query in query_list:
        print(f"\n=== 쿼리: '{query}' (기대 결과: {expected_result}) ===")

        # 기존 함수 사용하여 관련성 평가
        results = retrieve_and_evaluate_relevance(query, persist_directory)

        if not results:
            print("검색된 청크가 없거나 평가 중 오류가 발생했습니다.")
            continue

        # 결과 분석
        for result in results:
            actual_relevance = result["evaluation"].get("relevance")

            # 결과 일치 여부 확인
            result_matches = actual_relevance == expected_result

            # 결과 기록
            if result_matches:
                matching_evaluations += 1
                match_status = "일치 ✓"
            else:
                match_status = "불일치 ✗"

            total_evaluations += 1

            # 상세 결과 저장
            detailed_results.append({
                "query": query,
                "chunk_index": result["chunk_index"],
                "source": result["source"],
                "content_preview": result["content_preview"],
                "expected": expected_result,
                "actual": actual_relevance,
                "matches": result_matches
            })

            # 일치 여부 출력
            print(f"  청크 #{result['chunk_index'] + 1}: {match_status} (예상: {expected_result}, 실제: {actual_relevance})")

        # API 요청 간 간격 두기
        time.sleep(1)

    # 결과 요약
    if total_evaluations > 0:
        accuracy = (matching_evaluations / total_evaluations) * 100
        print(f"\n{expected_result} 쿼리 결과: {matching_evaluations}/{total_evaluations} 일치 ({accuracy:.2f}% 정확도)")
    else:
        print(f"\n{expected_result} 쿼리에 대한 평가 결과가 없습니다.")

    return matching_evaluations, total_evaluations, detailed_results


def run_complete_evaluation_test(yes_query_list, no_query_list):
    """
    모든 테스트 케이스에 대한 평가를 실행하고 결과를 분석합니다.
    """
    print("=== 테스트 시작: 'YES' 관련성이 예상되는 쿼리 ===")
    yes_matches, yes_total, yes_details = test_queries_for_expected_relevance(yes_query_list, "yes")

    if len(no_query_list) > 0:
        print("\n=== 테스트 시작: 'NO' 관련성이 예상되는 쿼리 ===")
        no_matches, no_total, no_details = test_queries_for_expected_relevance(no_query_list, "no")

    # 종합 결과 계산
    total_matches = yes_matches + no_matches
    total_evaluations = yes_total + no_total

    if total_evaluations > 0:
        overall_accuracy = (total_matches / total_evaluations) * 100
    else:
        overall_accuracy = 0

    # 종합 결과 출력
    print("\n=== 종합 평가 결과 ===")
    print(
        f"'yes' 케이스: {yes_matches}/{yes_total} 일치 ({yes_matches / yes_total * 100:.2f}% 정확도)" if yes_total > 0 else "'yes' 케이스: 평가 없음")
    print(
        f"'no' 케이스: {no_matches}/{no_total} 일치 ({no_matches / no_total * 100:.2f}% 정확도)" if no_total > 0 else "'no' 케이스: 평가 없음")
    print(f"전체: {total_matches}/{total_evaluations} 일치 ({overall_accuracy:.2f}% 정확도)")

    # 불일치 항목 분석
    print("\n=== 불일치 항목 분석 ===")
    mismatches = [r for r in yes_details + no_details if not r["matches"]]

    if mismatches:
        print(f"{len(mismatches)}개의 불일치 항목이 있습니다:")
        for i, mismatch in enumerate(mismatches):
            print(f"\n불일치 #{i + 1}:")
            print(f"쿼리: {mismatch['query']}")
            print(f"출처: {mismatch['source']}")
            print(f"내용 미리보기: {mismatch['content_preview']}")
            print(f"기대 결과: {mismatch['expected']}")
            print(f"실제 결과: {mismatch['actual']}")
    else:
        print("모든 평가 결과가 기대 결과와 일치합니다!")

    # 결과 반환
    return {
        "yes_results": yes_details,
        "no_results": no_details,
        "yes_accuracy": yes_matches / yes_total if yes_total > 0 else 0,
        "no_accuracy": no_matches / no_total if no_total > 0 else 0,
        "total_accuracy": overall_accuracy / 100 if total_evaluations > 0 else 0,
        "mismatches": mismatches
    }

if __name__ == "__main__":
    # 처리할 URL 목록
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    all_docs = load(urls)
    chunks = split_documents(all_docs)
    vectorstore = create_vector_store(chunks)

    #user_query = 'agent memory'
    #relevant_docs = retrieve_relevant_chunks(query=user_query)
    #evaluation_results = retrieve_and_evaluate_relevance(user_query)

    yes_query_list = [
        "LLM 기반 자율 에이전트 시스템의 계획, 메모리, 도구 사용 구성 요소에 대해 설명하고, 이를 개발 워크플로우에 어떻게 통합할 수 있을까요?",
        "Chain-of-Thought와 같은 프롬프트 엔지니어링 기법이 복잡한 추론 작업에서 LLM의 성능을 어떻게 향상시키는지 설명해주세요.",
        "LLM에 대한 적대적 공격 유형(토큰 조작, 그래디언트 기반 공격, 탈옥 프롬프팅 등)의 작동 방식과 이에 대한 방어 전략은 무엇인가요?"
    ]
    no_query_list = [
        "맨체스터 유나이티드의 2023-2024 시즌 성적과 주요 선수들의 활약상에 대해 분석해주세요.",
        "백종원 셰프의 간단한 김치찌개 레시피와 비법을 알려주세요.",
        "중세 유럽 건축양식의 변천사와 고딕 양식의 특징에 대해 설명해주세요."
    ]
    print("*========================================================================*")
    run_complete_evaluation_test(yes_query_list, no_query_list)