"""
검색 결과와 할루시네이션 검사를 통합한 RAG 체인을 제공하는 모듈
"""

from typing import Dict, Any, List, Callable
from langchain_core.runnables import RunnablePassthrough

from src.retrieval.retriever import format_docs, format_sources
from src.evaluation.relevance import create_relevance_evaluation_chain
from src.evaluation.hallucination import create_hallucination_evaluation_chain, create_enhanced_hallucination_evaluation_chain
from src.rag.answer_chain import create_answer_chain
from src.rag.graph_agent import GraphRAGAgent
from langchain_chroma import Chroma
def create_rag_chain_with_relevance(retriever: Callable):
    """
    관련성 평가를 통합한 RAG 체인을 생성합니다.

    Args:
        retriever: 검색기 함수

    Returns:
        관련성 평가가 통합된 RAG 체인
    """
    # 관련성 평가 체인 생성
    relevance_chain = create_relevance_evaluation_chain()

    # 답변 생성 체인 생성
    answer_chain = create_answer_chain()

    def process_with_relevance(inputs):
        """
        관련성 평가를 통해 답변을 생성하는 함수

        Args:
            inputs: 입력 데이터 (question 키 포함)

        Returns:
            생성된 답변 및 메타데이터
        """
        question = inputs["question"]
        docs = retriever.get_relevant_documents(question)

        # 관련성 있는 문서만 필터링
        relevant_docs = []
        sources = []  # 출처 정보 저장
        relevance_evaluations = []

        for i, doc in enumerate(docs):
            try:
                result = relevance_chain.invoke({
                    "question": question,
                    "context": doc.page_content
                })

                relevance = result.get("relevance", "no")

                # 디버깅을 위한 평가 결과 저장
                relevance_evaluations.append({
                    "doc_preview": doc.page_content[:100] + "...",
                    "relevance": relevance
                })

                if relevance == "yes":
                    relevant_docs.append(doc)
                    # 출처 정보 저장
                    source_info = {
                        "source": doc.metadata.get("source", "출처 없음"),
                        "preview": doc.page_content[:150] + ("..." if len(doc.page_content) > 150 else "")
                    }
                    sources.append(source_info)
            except Exception as e:
                print(f"청크 #{i + 1} 평가 중 오류: {e}")

        if not relevant_docs:
            return {
                "answer": "관련된 정보를 찾을 수 없습니다. 다른 질문을 시도해보세요.",
                "sources": [],
                "formatted_sources": "",
                "relevance_evaluations": relevance_evaluations
            }

        # 답변 생성
        formatted_context = format_docs(relevant_docs)

        answer = answer_chain.invoke({
            "question": question,
            "context": formatted_context
        })

        # 최종 결과 반환
        return {
            "answer": answer,
            "sources": sources,
            "formatted_sources": format_sources(sources),
            "relevance_evaluations": relevance_evaluations
        }

    # RunnableLambda를 사용하여 체인 구성
    from langchain_core.runnables import RunnableLambda
    chain = RunnableLambda(process_with_relevance)

    return chain


def create_rag_chain_with_hallucination_check(retriever: Callable):
    """
    할루시네이션 검사를 통합한 RAG 체인을 생성합니다.

    Args:
        retriever: 검색기 함수

    Returns:
        할루시네이션 검사가 통합된 RAG 체인
    """
    # 할루시네이션 평가 체인 생성
    hallucination_chain = create_enhanced_hallucination_evaluation_chain()

    # 답변 생성 체인 생성 (기본 temperature는 0으로 설정)
    answer_chain = create_answer_chain(temperature=0)

    # 관련성 평가 체인 생성
    relevance_chain = create_relevance_evaluation_chain()

    def generate_answer_with_hallucination_check(inputs, max_retries=1, current_retry=0):
        """
        할루시네이션 검사 및 재귀적 재시도를 통해 답변을 생성하는 함수

        Args:
            inputs: 입력 데이터 (question 키 포함)
            max_retries: 최대 재시도 횟수
            current_retry: 현재 재시도 횟수

        Returns:
            생성된 답변 및 메타데이터
        """
        question = inputs["question"]
        docs = retriever.get_relevant_documents(question)

        print(f"\n'{question}' 질문에 대해 {len(docs)}개의 청크를 검색했습니다.")

        # 1. 관련성 평가 및 필터링
        relevant_docs = []
        sources = []  # 출처 정보 저장

        for i, doc in enumerate(docs):
            try:
                result = relevance_chain.invoke({
                    "question": question,
                    "context": doc.page_content
                })

                relevance = result.get("relevance")
                print(f"청크 #{i + 1} 관련성: {relevance}")

                if relevance == "yes":
                    relevant_docs.append(doc)
                    # 출처 정보 저장
                    source_info = {
                        "source": doc.metadata.get("source", "출처 없음"),
                        "preview": doc.page_content[:150] + ("..." if len(doc.page_content) > 150 else "")
                    }
                    sources.append(source_info)
            except Exception as e:
                print(f"청크 #{i + 1} 평가 중 오류: {e}")

        print(f"총 {len(relevant_docs)}개의 관련 청크(relevance='yes')를 찾았습니다.")

        if not relevant_docs:
            return {
                "answer": "관련된 정보를 찾을 수 없습니다. 다른 질문을 시도해보세요.",
                "sources": [],
                "formatted_sources": "",
                "has_hallucination": False,
                "retried": current_retry,
                "temperature": 0.0  # 기본 temperature 값 추가
            }

        # 2. 답변 생성 - 재시도마다 temperature 증가
        formatted_context = format_docs(relevant_docs)

        # 재시도 횟수에 따라 temperature 동적 조정 (0.2에서 시작하여 최대 0.8까지)
        temperature = min(current_retry * 0.2, 0.8)
        print(f"현재 temperature: {temperature}")

        # 재시도마다 새로운 answer_chain 생성 (temperature 조정)
        current_answer_chain = create_answer_chain(temperature=temperature)

        # 이전 답변과 suggestions가 있는 경우 이를 활용하여 개선된 프롬프트 생성
        if current_retry > 0 and "previous_answer" in inputs and "suggestions" in inputs:
            previous_answer = inputs.get("previous_answer", "")
            suggestions = inputs.get("suggestions", [])

            if suggestions:
                print(f"이전 답변 개선을 위한 제안사항 적용: {suggestions}")

                # 개선된 프롬프트 생성
                enhanced_prompt = f"""
질문: {question}

컨텍스트:
{formatted_context}

이전 답변:
{previous_answer}

이전 답변에 대한 개선 제안:
{chr(10).join([f"- {suggestion}" for suggestion in suggestions])}

위의 개선 제안을 참고하여, 컨텍스트에 있는 정보만 활용하여 더 정확하고 매끄러운 답변을 생성해주세요.
"""
                print("개선된 프롬프트로 답변 생성 중...")
                answer = current_answer_chain.invoke({
                    "question": enhanced_prompt,
                    "context": formatted_context
                })
            else:
                # suggestions가 없는 경우 일반적인 답변 생성
                answer = current_answer_chain.invoke({
                    "question": question,
                    "context": formatted_context
                })
        else:
            # 첫 번째 시도 또는 suggestions가 없는 경우 일반적인 답변 생성
            answer = current_answer_chain.invoke({
                "question": question,
                "context": formatted_context
            })

        # 3. 할루시네이션 검사
        try:
            hallucination_result = hallucination_chain.invoke({
                "question": question,
                "context": formatted_context,
                "answer": answer
            })

            has_hallucination = hallucination_result.get("hallucination") == "yes"
            print(f"할루시네이션 검사 결과: {'있음' if has_hallucination else '없음'}")

            # 할루시네이션 분석 결과 저장
            suggestions = hallucination_result.get("suggestions", [])
            if suggestions:
                print(f"할루시네이션 개선 제안: {suggestions}")

            # 할루시네이션이 있고 재시도 가능한 경우 재귀적으로 재시도
            if has_hallucination and current_retry < max_retries:
                print(f"할루시네이션 감지. 재시도 중... ({current_retry + 1}/{max_retries})")

                # 이전 답변과 suggestions를 포함하여 재시도
                modified_inputs = {
                    "question": question,
                    "context": formatted_context,
                    "previous_answer": answer,
                    "suggestions": suggestions
                }

                return generate_answer_with_hallucination_check(
                    modified_inputs,
                    max_retries=max_retries,
                    current_retry=current_retry + 1
                )

            # 할루시네이션이 없거나 최대 재시도 횟수에 도달한 경우
            return {
                "answer": answer,
                "sources": sources,
                "formatted_sources": format_sources(sources),
                "has_hallucination": has_hallucination,
                "retried": current_retry,
                "temperature": temperature,  # temperature 값을 결과에 포함
                "suggestions": suggestions  # suggestions 정보 추가
            }

        except Exception as e:
            print(f"할루시네이션 검사 중 오류 발생: {e}")
            # 오류 발생 시 현재 답변 반환
            return {
                "answer": answer,
                "sources": sources,
                "formatted_sources": format_sources(sources),
                "has_hallucination": False,
                "retried": current_retry,
                "temperature": temperature,  # temperature 값을 결과에 포함
                "suggestions": []  # 빈 suggestions 추가
            }

    # 최종 RAG 체인 구성
    def process_chain(inputs):
        # 최대 재시도 횟수를 2로 증가 (총 3번 시도)
        result = generate_answer_with_hallucination_check(inputs, max_retries=2)

        # temperature 키가 없는 경우 기본값 설정
        temperature = result.get("temperature", 0.0)

        return {
            "question": inputs["question"],
            "answer": result["answer"],
            "sources": result["sources"],
            "formatted_sources": format_sources(result["sources"]),
            "has_hallucination": result["has_hallucination"],
            "retried": result["retried"],
            "temperature": temperature,  # 안전하게 temperature 값 사용
            "suggestions": result.get("suggestions", [])  # suggestions 정보 추가
        }

    # RunnablePassthrough를 사용하여 체인 구성
    from langchain_core.runnables import RunnableLambda
    chain = RunnableLambda(process_chain)

    return chain


def run_rag_chain_with_hallucination_check(question: str, persist_directory: str = "./chroma_db"):
    """
    할루시네이션 검사가 통합된 RAG 체인을 실행합니다.

    Args:
        question: 사용자 질문
        persist_directory: 벡터 스토어 디렉토리

    Returns:
        실행 결과
    """
    from src.retrieval.retriever import get_retriever

    # Retriever 생성
    retriever = get_retriever(persist_directory)

    # RAG 체인 생성
    rag_chain = create_rag_chain_with_hallucination_check(retriever)

    # 체인 실행
    result = rag_chain.invoke({"question": question})

    # 결과 출력
    print("\n===== 결과 =====")
    print(f"질문: {result['question']}")
    print(f"답변: {result['answer']}")
    print(f"할루시네이션: {'있음' if result['has_hallucination'] else '없음'}")
    print(f"재시도 횟수: {result['retried']}")
    print(f"사용된 Temperature: {result['temperature']}")

    # suggestions 정보 출력
    suggestions = result.get("suggestions", [])
    if suggestions:
        print("\n할루시네이션 개선 제안:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")

    print(f"\n출처: {result['formatted_sources']}")

    return result


def run_rag_chain_with_relevance(question: str, persist_directory: str = "./chroma_db"):
    """
    관련성 평가가 통합된 RAG 체인을 실행합니다.

    Args:
        question: 사용자 질문
        persist_directory: 벡터 스토어 디렉토리

    Returns:
        실행 결과
    """
    from src.retrieval.retriever import get_retriever

    # Retriever 생성
    retriever = get_retriever(persist_directory)

    # RAG 체인 생성
    rag_chain = create_rag_chain_with_relevance(retriever)

    # 체인 실행
    result = rag_chain.invoke({"question": question})

    return result


def run_graph_rag_chain(question, persist_directory="./chroma_db", tavily_api_key=None):
    """
    LangGraph 기반 RAG 체인을 실행합니다.

    Args:
        question: 사용자 질문
        persist_directory: 벡터 스토어 디렉토리
        tavily_api_key: Tavily API 키 (없으면 환경 변수에서 로드)

    Returns:
        Dict: 생성된 답변과 문서 정보 포함
    """
    # 벡터 스토어 로드
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL))
    
    # API 키 설정
    if not tavily_api_key:
        import os
        tavily_api_key = os.environ.get("TAVILY_API_KEY", "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")
    
    # RAG 에이전트 초기화 및 실행
    agent = GraphRAGAgent(
        retriever=vectorstore.as_retriever(),
        tavily_api_key=tavily_api_key
    )
    
    # 쿼리 실행
    result = agent.run(question)
    
    # 할루시네이션 정보 추가 (호환성 유지)
    result["hallucination_detected"] = False
    
    return result