"""
RAPTOR를 사용한 RAG 체인 모듈 - 할루시네이션 개선 버전
"""

from typing import Dict, Any, List
from src.raptor.raptor_index import get_raptor_retriever
from src.retrieval.retriever import format_docs, format_sources
from src.evaluation.hallucination import create_hallucination_evaluation_chain, create_enhanced_hallucination_evaluation_chain
from src.rag.answer_chain import create_answer_chain


def create_raptor_rag_chain_with_hallucination_check(retriever, enable_retry: bool = True):
    """
    RAPTOR 인덱스를 사용하고 할루시네이션 검사를 수행하는 RAG 체인을 생성합니다.

    Args:
        retriever: RAPTOR retriever
        enable_retry: 재시도 로직 활성화 여부 (기본값: True)

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
        context = inputs.get("context", "")
        formatted_context = context
        
        current_answer_chain = answer_chain
        
        # 할루시네이션 분석 처리 및 응답 생성 로직
        if current_retry > 0 and "previous_answer" in inputs and "hallucination_analysis" in inputs:
            previous_answer = inputs.get("previous_answer", "")
            hallucination_analysis = inputs.get("hallucination_analysis", {})
            suggestions = hallucination_analysis.get("suggestions", [])
            problematic_parts = hallucination_analysis.get("problematic_parts", [])
            
            # 할루시네이션 분석 결과를 확인
            if hallucination_analysis.get("hallucination", "no") == "yes":
                print(f"할루시네이션 감지됨: {hallucination_analysis.get('analysis', '분석 정보 없음')}")
                
                # 할루시네이션 문제가 심각한 경우
                if len(problematic_parts) > 0 and len(suggestions) > 0:
                    # 사용자에게 할루시네이션 정보를 알리고 수정된 답변 제공
                    answer = f"""제시하신 질문에 대한 답변을 생성하는 과정에서 확인된 정보와 맞지 않는 내용이 포함될 수 있어 알려드립니다.

{hallucination_analysis.get('analysis', '문서에 명시되지 않은 내용이 포함될 수 있습니다.')}

제공된 문서 내용에 기반하여 정확한 정보를 알려드리면:

"""
                    # 개선된 프롬프트 생성하여 정확한 답변 생성
                    enhanced_prompt = f"""
질문: {question}

컨텍스트:
{formatted_context}

다음 사항에 주의하여 답변을 생성해주세요:
1. 컨텍스트에 명시적으로 포함된 정보만 사용할 것
2. 확실하지 않은 정보는 '~일 수 있습니다', '문서에 언급되지 않았습니다' 등으로 표현할 것
3. 추측성 정보는 제공하지 말 것

위 지침을 참고하여, 컨텍스트에 있는 정보만 활용하여 정확한 답변을 생성해주세요.
"""
                    factual_answer = current_answer_chain.invoke({
                        "question": enhanced_prompt,
                        "context": formatted_context
                    })
                    
                    # 최종 답변 조합
                    answer += factual_answer
                else:
                    # 할루시네이션이 감지되었지만 구체적인 문제점이나 제안이 없는 경우
                    answer = f"""제시하신 질문에 대해 정확한 답변을 드리기 어렵습니다. 제공된 문서에서 관련 정보를 명확히 확인할 수 없습니다.

현재 문서에 포함된 정보만을 기반으로 답변드립니다:

"""
                    simplified_prompt = f"제공된 문서 내용만을 기반으로 '{question}'에 대한 답변을 간략히 작성해주세요. 확실하지 않은 정보는 언급하지 마세요."
                    basic_answer = current_answer_chain.invoke({
                        "question": simplified_prompt,
                        "context": formatted_context
                    })
                    
                    answer += basic_answer
            else:
                # 할루시네이션이 없는 경우 일반 답변 생성
                answer = current_answer_chain.invoke({
                    "question": question,
                    "context": formatted_context
                })
        else:
            # 첫 번째 시도인 경우 일반적인 답변 생성
            answer = current_answer_chain.invoke({
                "question": question,
                "context": formatted_context
            })
        
        # 최초 시도일 경우 할루시네이션 검사 수행
        if current_retry == 0:
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
            
            # 할루시네이션이 있고 재시도 가능한 경우 (재시도 로직이 활성화된 경우에만)
            if has_hallucination and current_retry < max_retries and enable_retry:
                print(f"할루시네이션 감지. 재시도 중... ({current_retry + 1}/{max_retries})")
                # 재시도 로직 실행
                return generate_answer_with_raptor({
                    "question": question,
                    "context": context,
                    "previous_answer": answer,
                    "hallucination_analysis": enhanced_result
                }, max_retries, current_retry + 1)
            
            # 최종 결과 반환
            return {
                "answer": answer,
                "hallucination_detected": has_hallucination,
                "hallucination_analysis": enhanced_result,
                "retry_count": current_retry,
                "suggestions": enhanced_result.get("suggestions", [])
            }
        else:
            # 재시도 결과 반환
            return {
                "answer": answer,
                "hallucination_detected": inputs.get("hallucination_analysis", {}).get("hallucination", "no") == "yes",
                "hallucination_analysis": inputs.get("hallucination_analysis", {}),
                "retry_count": current_retry,
                "suggestions": inputs.get("hallucination_analysis", {}).get("suggestions", [])
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


def run_raptor_rag_chain(question: str, persist_directory: str = "./raptor_index", enable_retry: bool = True):
    """
    RAPTOR RAG 체인을 실행합니다.

    Args:
        question: 사용자 질문
        persist_directory: RAPTOR 인덱스가 저장된 디렉토리
        enable_retry: 재시도 로직 활성화 여부 (기본값: True)

    Returns:
        실행 결과
    """
    # RAPTOR Retriever 생성
    retriever = get_raptor_retriever(persist_directory)
    
    # RAPTOR RAG 체인 생성
    rag_chain = create_raptor_rag_chain_with_hallucination_check(retriever, enable_retry=enable_retry)
    
    # 체인 실행
    result = rag_chain.invoke({"question": question})
    
    # 결과 출력
    print("\n===== 결과 =====")
    print(f"질문: {question}")
    print(f"답변: {result['answer']}")
    print(f"할루시네이션: {'있음' if result['hallucination_detected'] else '없음'}")
    print(f"재시도 횟수: {result['retry_count']}")
    
    # suggestions 정보 출력
    suggestions = result.get("suggestions", [])
    if suggestions:
        print("\n할루시네이션 개선 제안:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
    
    print(f"\n출처: {result['formatted_sources']}")
    
    return result 