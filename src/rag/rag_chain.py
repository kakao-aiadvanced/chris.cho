"""
검색 결과와 할루시네이션 검사를 통합한 RAG 체인을 제공하는 모듈
"""

from typing import Dict, Any, List, Callable
from langchain_core.runnables import RunnablePassthrough

from src.retrieval.retriever import format_docs, format_sources
from src.evaluation.relevance import create_relevance_evaluation_chain
from src.evaluation.hallucination import create_hallucination_evaluation_chain, create_enhanced_hallucination_evaluation_chain
from src.rag.answer_chain import create_answer_chain

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
    
    # 답변 생성 체인 생성
    answer_chain = create_answer_chain()

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
                "retried": current_retry
            }

        # 2. 답변 생성
        formatted_context = format_docs(relevant_docs)

        answer = answer_chain.invoke({
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

            # 할루시네이션이 있고 재시도 가능한 경우 재귀적으로 재시도
            if has_hallucination and current_retry < max_retries:
                print(f"할루시네이션 감지. 재시도 중... ({current_retry + 1}/{max_retries})")
                return generate_answer_with_hallucination_check(
                    inputs,
                    max_retries=max_retries,
                    current_retry=current_retry + 1
                )

            # 할루시네이션이 없거나 최대 재시도 횟수에 도달한 경우
            return {
                "answer": answer,
                "sources": sources,
                "formatted_sources": format_sources(sources),
                "has_hallucination": has_hallucination,
                "retried": current_retry
            }

        except Exception as e:
            print(f"할루시네이션 검사 중 오류 발생: {e}")
            # 오류 발생 시 현재 답변 반환
            return {
                "answer": answer,
                "sources": sources,
                "formatted_sources": format_sources(sources),
                "has_hallucination": False,
                "retried": current_retry
            }

    # 최종 RAG 체인 구성
    def process_chain(inputs):
        result = generate_answer_with_hallucination_check(inputs, max_retries=1)
        return {
            "question": inputs["question"],
            "answer": result["answer"],
            "sources": result["sources"],
            "formatted_sources": format_sources(result["sources"]),
            "has_hallucination": result["has_hallucination"],
            "retried": result["retried"]
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