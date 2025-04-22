"""
RAPTOR를 사용한 RAG 체인 모듈
"""

from typing import Dict, Any
from src.raptor.raptor_index import get_raptor_retriever
from src.retrieval.retriever import format_docs, format_sources
from src.evaluation.hallucination import create_hallucination_evaluation_chain, create_enhanced_hallucination_evaluation_chain
from src.rag.answer_chain import create_answer_chain

def create_raptor_rag_chain_with_hallucination_check(retriever):
    """
    RAPTOR 인덱스를 사용하고 할루시네이션 검사를 수행하는 RAG 체인을 생성합니다.

    Args:
        retriever: RAPTOR retriever

    Returns:
        RAPTOR RAG 체인
    """
    # 할루시네이션 평가 체인 생성
    hallucination_chain = create_hallucination_evaluation_chain()
    enhanced_hallucination_chain = create_enhanced_hallucination_evaluation_chain()
    
    # 답변 생성 체인 생성
    answer_chain = create_answer_chain()

    def generate_answer_with_raptor(inputs: Dict[str, Any], max_retries: int = 1, current_retry: int = 0) -> Dict[str, Any]:
        """
        RAPTOR 검색 결과를 사용하여 답변을 생성하고 할루시네이션 검사를 수행합니다.

        Args:
            inputs: 입력 사전 (question, context)
            max_retries: 최대 재시도 횟수
            current_retry: 현재 재시도 횟수

        Returns:
            처리 결과
        """
        question = inputs["question"]
        context = inputs["context"]
        
        # 답변 생성
        answer = answer_chain.invoke({"question": question, "context": context})
        
        # 할루시네이션 검사
        hallucination_result = hallucination_chain.invoke({
            "question": question,
            "context": context,
            "answer": answer
        })
        
        # 상세 할루시네이션 분석
        enhanced_result = enhanced_hallucination_chain.invoke({
            "question": question,
            "context": context,
            "answer": answer
        })
        
        has_hallucination = hallucination_result.get("hallucination", "no") == "yes"
        
        # 할루시네이션이 있고 재시도 가능한 경우
        if has_hallucination and current_retry < max_retries:
            # 재시도용 프롬프트 생성
            retry_prompt = f"""
질문: {question}

컨텍스트:
{context}

이전 답변:
{answer}

할루시네이션 분석:
{enhanced_result.get('analysis', '분석 정보 없음')}

문제가 된 부분:
{', '.join(enhanced_result.get('problematic_parts', ['문제 부분 정보 없음']))}

개선 제안:
{', '.join(enhanced_result.get('suggestions', ['제안 정보 없음']))}

위의 분석을 참고하여, 컨텍스트에 있는 정보만 활용하여 정확한 답변을 다시 생성해주세요.
            """
            
            # 재시도 답변 생성
            retry_answer = answer_chain.invoke({"question": retry_prompt, "context": context})
            
            # 재귀적으로 할루시네이션 검사 수행
            return generate_answer_with_raptor({
                "question": question,
                "context": context,
                "retry_prompt": retry_prompt,
                "previous_answer": answer
            }, max_retries, current_retry + 1)
        
        # 최종 결과 반환
        return {
            "answer": answer,
            "hallucination_detected": has_hallucination,
            "hallucination_analysis": enhanced_result,
            "retry_count": current_retry
        }

    # RAG 체인 구성
    rag_chain = (
        {
            "question": lambda x: x["question"],
            "docs": lambda x: retriever.get_relevant_documents(x["question"])
        }
        | {
            "question": lambda x: x["question"],
            "context": lambda x: format_docs(x["docs"]) if x["docs"] else "관련 정보를 찾을 수 없습니다.",
            "sources": lambda x: x["docs"]
        }
        | {
            **generate_answer_with_raptor,
            "sources": lambda x: x["sources"],
            "formatted_sources": lambda x: format_sources(x["sources"])
        }
    )

    return rag_chain


def run_raptor_rag_chain(question: str, persist_directory: str = "./raptor_index"):
    """
    RAPTOR RAG 체인을 실행합니다.

    Args:
        question: 사용자 질문
        persist_directory: RAPTOR 인덱스가 저장된 디렉토리

    Returns:
        실행 결과
    """
    # RAPTOR Retriever 생성
    retriever = get_raptor_retriever(persist_directory)
    
    # RAPTOR RAG 체인 생성
    rag_chain = create_raptor_rag_chain_with_hallucination_check(retriever)
    
    # 체인 실행
    result = rag_chain.invoke({"question": question})
    
    return result 