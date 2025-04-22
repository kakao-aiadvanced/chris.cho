"""
RAG 시스템의 벤치마킹 및 테스트를 위한 유틸리티 모듈
"""

import time
from typing import List, Dict, Any, Tuple


def test_queries_for_expected_relevance(query_list: List[str], expected_result: str,
                                        persist_directory: str = "./chroma_db") -> Tuple[int, int, List[Dict]]:
    """
    쿼리 목록에 대해 예상된 관련성을 가진 검색 결과를 테스트합니다.

    Args:
        query_list: 테스트할 쿼리 목록
        expected_result: 예상되는 관련성 결과 ("yes" 또는 "no")
        persist_directory: 벡터 스토어 디렉토리

    Returns:
        (성공 횟수, 총 쿼리 수, 상세 결과 목록) 튜플
    """
    from src.retrieval.retriever import get_retriever
    from src.evaluation.relevance import create_relevance_evaluation_chain
    
    # 검색기 및 관련성 평가 체인 생성
    retriever = get_retriever(persist_directory)
    relevance_chain = create_relevance_evaluation_chain()
    
    # 결과 추적
    success_count = 0
    total_queries = len(query_list)
    detailed_results = []
    
    print(f"총 {total_queries}개의 '{expected_result}' 쿼리 테스트 시작...\n")
    
    for idx, query in enumerate(query_list):
        print(f"[{idx+1}/{total_queries}] 쿼리: '{query}' 테스트 중...")
        
        # 관련 청크 검색
        start_time = time.time()
        docs = retriever.get_relevant_documents(query)
        retrieval_time = time.time() - start_time
        
        # 첫 번째 검색 결과에 대한 관련성 평가
        relevance_evaluations = []
        for doc in docs:
            result = relevance_chain.invoke({
                "question": query,
                "context": doc.page_content
            })
            relevance = result.get("relevance", "no")
            relevance_evaluations.append({
                "relevance": relevance,
                "doc_preview": doc.page_content[:100] + "..."
            })
        
        # 최소 하나 이상의 관련 결과를 찾았는지 여부
        found_relevant = any(eval["relevance"] == "yes" for eval in relevance_evaluations)
        
        # 예상 결과와 실제 결과가 일치하는지 확인
        success = (found_relevant and expected_result == "yes") or (not found_relevant and expected_result == "no")
        success_icon = "✓" if success else "✗"
        
        if success:
            success_count += 1
        
        # 결과 저장
        detailed_results.append({
            "query": query,
            "expected": expected_result,
            "found_relevant": found_relevant,
            "success": success,
            "retrieval_time": retrieval_time,
            "evaluation_details": relevance_evaluations
        })
        
        # 결과 출력
        print(f"  결과: {success_icon} {'성공' if success else '실패'} (예상: {expected_result}, 실제: {'yes' if found_relevant else 'no'})")
        print(f"  검색 시간: {retrieval_time:.2f}초")
        print()
    
    # 종합 결과 출력
    success_rate = (success_count / total_queries) * 100
    print(f"테스트 완료: {success_count}/{total_queries} ({success_rate:.1f}%) 성공\n")
    
    return success_count, total_queries, detailed_results


def run_complete_evaluation_test(yes_query_list: List[str], no_query_list: List[str], persist_directory: str = "./chroma_db"):
    """
    관련성이 있어야 하는 쿼리와 관련성이 없어야 하는 쿼리 모두에 대해 종합적인 평가 테스트를 실행합니다.

    Args:
        yes_query_list: 관련성이 있어야 하는 쿼리 목록
        no_query_list: 관련성이 없어야 하는 쿼리 목록
        persist_directory: 벡터 스토어 디렉토리

    Returns:
        종합 테스트 결과 사전
    """
    print("=== 종합 평가 테스트 ===")
    
    # "yes" 쿼리 테스트
    print("\n[관련성 있음 쿼리 테스트]")
    yes_success, yes_total, yes_details = test_queries_for_expected_relevance(
        yes_query_list, "yes", persist_directory
    )
    
    # "no" 쿼리 테스트
    print("\n[관련성 없음 쿼리 테스트]")
    no_success, no_total, no_details = test_queries_for_expected_relevance(
        no_query_list, "no", persist_directory
    )
    
    # 종합 메트릭 계산
    total_success = yes_success + no_success
    total_queries = yes_total + no_total
    overall_success_rate = (total_success / total_queries) * 100
    
    # 정밀도(Precision)와 재현율(Recall)과 유사한 메트릭 계산
    true_positives = yes_success  # 관련성 있는 것을 관련성 있다고 올바르게 식별
    false_positives = no_total - no_success  # 관련성 없는 것을 관련성 있다고 잘못 식별
    false_negatives = yes_total - yes_success  # 관련성 있는 것을 관련성 없다고 잘못 식별
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 종합 결과 출력
    print("\n=== 종합 결과 ===")
    print(f"총 테스트 쿼리: {total_queries}")
    print(f"총 성공: {total_success}/{total_queries} ({overall_success_rate:.1f}%)")
    print(f"관련성 있음 쿼리 성공율: {yes_success}/{yes_total} ({(yes_success/yes_total*100):.1f}%)")
    print(f"관련성 없음 쿼리 성공율: {no_success}/{no_total} ({(no_success/no_total*100):.1f}%)")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # 결과 반환
    return {
        "total_queries": total_queries,
        "total_success": total_success,
        "overall_success_rate": overall_success_rate,
        "yes_success": yes_success,
        "yes_total": yes_total,
        "no_success": no_success,
        "no_total": no_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_details": yes_details,
        "no_details": no_details
    } 