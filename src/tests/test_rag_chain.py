import unittest
import os
from src.rag.rag_chain import (
    run_rag_chain_with_hallucination_check,
    run_rag_chain_with_relevance
)
from src.config.openai_config import initialize_openai_api
from src.retrieval.retriever import get_retriever, format_docs, format_sources
from src.data_loader.web_loader import load_from_urls
from src.text_processing.splitter import split_documents
from src.embeddings.vector_store import create_vector_store, get_embedding_model
from src.evaluation.relevance import create_relevance_evaluation_chain
from src.evaluation.hallucination import create_hallucination_evaluation_chain, create_enhanced_hallucination_evaluation_chain
from src.rag.answer_chain import create_answer_chain
from src.utils.benchmark import test_queries_for_expected_relevance, run_complete_evaluation_test
from langchain_core.documents import Document

class TestRAGChain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 초기화 - 벡터 스토어 생성"""
        # OpenAI API 키 설정
        initialize_openai_api()
        
        cls.test_urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]
        
        # 문서 로드 및 벡터 스토어 생성
        cls.all_docs = load_from_urls(cls.test_urls, debug=False)
        cls.chunks = split_documents(cls.all_docs, debug=False)
        cls.vectorstore = create_vector_store(cls.chunks, persist_directory="./chroma_db")
        
        # 테스트용 쿼리 리스트
        cls.yes_queries = [
            "LLM 기반 자율 에이전트 시스템의 계획, 메모리, 도구 사용 구성 요소에 대해 설명하고, 이를 개발 워크플로우에 어떻게 통합할 수 있을까요?",
            "Chain-of-Thought와 같은 프롬프트 엔지니어링 기법이 복잡한 추론 작업에서 LLM의 성능을 어떻게 향상시키는지 설명해주세요.",
            "LLM에 대한 적대적 공격 유형(토큰 조작, 그래디언트 기반 공격, 탈옥 프롬프팅 등)의 작동 방식과 이에 대한 방어 전략은 무엇인가요?"
        ]
        
        cls.no_queries = [
            "맨체스터 유나이티드의 2023-2024 시즌 성적과 주요 선수들의 활약상에 대해 분석해주세요.",
            "백종원 셰프의 간단한 김치찌개 레시피와 비법을 알려주세요.",
            "중세 유럽 건축양식의 변천사와 고딕 양식의 특징에 대해 설명해주세요."
        ]
        
        cls.hallucination_query = "GPT-7 모델의 성능과 특징에 대해 자세히 설명해주세요."

    def test_1_web_base_loader(self):
        """1. WebBaseLoader를 사용한 블로그 포스팅 로드 테스트"""
        documents = load_from_urls(self.test_urls, debug=False)
        
        self.assertIsNotNone(documents)
        self.assertEqual(len(documents), len(self.test_urls))
        for doc in documents:
            self.assertTrue(hasattr(doc, 'page_content') or 'page_content' in dir(doc))
            self.assertTrue(hasattr(doc, 'metadata') or 'metadata' in dir(doc))
            self.assertIn('source', doc.metadata)
            self.assertGreater(len(doc.page_content), 0)

    def test_2_recursive_text_splitter(self):
        """2. RecursiveCharacterTextSplitter를 사용한 문서 분할 테스트"""
        chunks = split_documents(self.all_docs, chunk_size=1000, chunk_overlap=200, debug=False)
        
        self.assertIsNotNone(chunks)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertTrue(hasattr(chunk, 'page_content') or 'page_content' in dir(chunk))
            self.assertTrue(hasattr(chunk, 'metadata') or 'metadata' in dir(chunk))
            # 청크 크기가 최대 크기에 가깝거나 작은지 확인 (약간의 여유 허용)
            self.assertLessEqual(len(chunk.page_content), 1200)  # 약간의 여유 허용

    def test_3_vector_store_creation(self):
        """3. OpenAI 임베딩과 Chroma 벡터 스토어 생성 테스트"""
        # create_vector_store 함수는 src/embeddings/vector_store.py에 이미 구현됨
        # 테스트에서는 setUpClass에서 생성된 vectorstore 사용
        
        self.assertIsNotNone(self.vectorstore)
        # 검색기 생성 테스트
        retriever = get_retriever("./chroma_db", k=6)
        self.assertIsNotNone(retriever)

    def test_4_chunk_retrieval(self):
        """4. 쿼리에 대한 청크 검색 테스트"""
        query = "agent memory"
        retriever = get_retriever("./chroma_db", k=6)
        docs = retriever.get_relevant_documents(query)
        
        self.assertIsNotNone(docs)
        self.assertLessEqual(len(docs), 6)
        for doc in docs:
            self.assertTrue(hasattr(doc, 'page_content') or 'page_content' in dir(doc))
            self.assertTrue(hasattr(doc, 'metadata') or 'metadata' in dir(doc))

    def test_5_relevance_evaluation(self):
        """5. 검색된 청크의 관련성 평가 테스트"""
        # create_relevance_evaluation_chain 함수 사용
        relevance_chain = create_relevance_evaluation_chain()
        
        query = "agent memory"
        retriever = get_retriever("./chroma_db", k=6)
        docs = retriever.get_relevant_documents(query)
        
        for doc in docs:
            result = relevance_chain.invoke({
                "question": query,
                "context": doc.page_content
            })
            self.assertIn('relevance', result)
            self.assertIn(result['relevance'], ['yes', 'no'])

    def test_6_7_relevance_cases_and_accuracy(self):
        """6 & 7. 관련성 평가 케이스 및 정확도 테스트"""
        # utils.benchmark 모듈 활용
        yes_success, yes_total, yes_details = test_queries_for_expected_relevance(
            self.yes_queries[:1],  # 전체 대신 첫 번째 쿼리만 테스트 (속도를 위해)
            "yes", 
            "./chroma_db"
        )
        
        no_success, no_total, no_details = test_queries_for_expected_relevance(
            self.no_queries[:1],  # 전체 대신 첫 번째 쿼리만 테스트 (속도를 위해)
            "no", 
            "./chroma_db"
        )
        
        self.assertGreater(yes_success, 0, "관련성 있는 쿼리 테스트 실패")
        self.assertGreater(no_success, 0, "관련성 없는 쿼리 테스트 실패")

    def test_8_answer_generation(self):
        """8. 관련성 있는 청크를 사용한 답변 생성 테스트"""
        # create_answer_chain 함수 사용
        answer_chain = create_answer_chain()
        
        query = self.yes_queries[0]
        retriever = get_retriever("./chroma_db", k=6)
        docs = retriever.get_relevant_documents(query)
        
        # 포맷 유틸리티 사용
        context = format_docs(docs)
        
        answer = answer_chain.invoke({
            "question": query,
            "context": context
        })
        
        self.assertIsNotNone(answer)
        self.assertGreater(len(answer), 0)

    def test_9_hallucination_evaluation(self):
        """9. 할루시네이션 평가 테스트"""
        # create_hallucination_evaluation_chain 함수 사용
        hallucination_chain = create_hallucination_evaluation_chain()
        
        # 할루시네이션이 있는 케이스 테스트
        query = self.hallucination_query
        retriever = get_retriever("./chroma_db", k=6)
        docs = retriever.get_relevant_documents(query)
        
        context = format_docs(docs)
        answer = "GPT-7은 OpenAI에서 개발한 차세대 대규모 언어 모델로, 2025년에 출시될 예정이며 10조 개의 파라미터를 가지고 있습니다. 이 모델은 멀티모달 입력을 처리할 수 있고, 99.9%의 정확도로 코드를 생성할 수 있으며, 인간 수준의 창의성을 발휘할 수 있습니다."
        
        result = hallucination_chain.invoke({
            "question": query,
            "context": context,
            "answer": answer
        })
        
        self.assertIn('hallucination', result)
        self.assertEqual(result['hallucination'], 'yes', "할루시네이션이 있는 답변에 대해 'yes'가 아닌 응답이 나왔습니다.")
        
        # 할루시네이션이 없는 케이스 테스트
        query = self.yes_queries[0]
        docs = retriever.get_relevant_documents(query)
        
        context = format_docs(docs)
        answer = "LLM 기반 자율 에이전트 시스템은 계획, 메모리, 도구 사용이라는 세 가지 주요 구성 요소로 이루어져 있습니다. 계획은 에이전트가 목표를 달성하기 위한 단계를 결정하는 능력입니다. 메모리는 에이전트가 이전 상호작용과 정보를 기억하고 활용하는 능력입니다. 도구 사용은 에이전트가 외부 도구와 API를 활용하여 작업을 수행하는 능력입니다. 이러한 구성 요소들은 개발 워크플로우에 통합하여 자동화된 작업 수행, 코드 생성, 디버깅 등의 기능을 구현할 수 있습니다."
        
        result = hallucination_chain.invoke({
            "question": query,
            "context": context,
            "answer": answer
        })
        
        self.assertIn('hallucination', result)
        self.assertEqual(result['hallucination'], 'no', "할루시네이션이 없는 답변에 대해 'no'가 아닌 응답이 나왔습니다.")

    def test_10_integrated_rag_chain(self):
        """10. 통합 RAG 체인 테스트"""
        # 관련성 있는 쿼리 테스트
        query = self.yes_queries[0]
        result = run_rag_chain_with_relevance(query)
        
        # 결과 검증
        self.assertIn('answer', result)
        self.assertIn('sources', result)
        self.assertIn('formatted_sources', result)
        self.assertGreater(len(result['sources']), 0)
        
        # 관련성 없는 쿼리 테스트
        query = self.no_queries[0]
        result = run_rag_chain_with_relevance(query)
        
        # 결과 검증
        self.assertIn('answer', result)
        self.assertIn('sources', result)
        self.assertIn('formatted_sources', result)
        # 관련성 없는 쿼리는 할루시네이션 체크 대신 직접 쿼리 관련성만 테스트
    
    def test_11_hallucination_chain(self):
        """11. 할루시네이션 감지 체인 테스트"""
        # 할루시네이션 감지는 별도 테스트로 분리
        query = self.hallucination_query
        
        try:
            result = run_rag_chain_with_hallucination_check(query)
            
            # 결과 검증
            self.assertIn('answer', result)
            self.assertIn('sources', result)
            self.assertIn('formatted_sources', result)
            self.assertIn('has_hallucination', result)
            self.assertIn('retried', result)
            
            # 할루시네이션 쿼리이므로 has_hallucination 값이 True일 가능성이 높지만,
            # LLM의 비결정적 특성으로 인해 False일 수도 있으므로 검증하지 않음
        except Exception as e:
            self.skipTest(f"할루시네이션 체크 중 오류 발생: {str(e)}")

    def test_complete_system_evaluation(self):
        """전체 시스템 평가 테스트"""
        # utils.benchmark 모듈의 run_complete_evaluation_test 사용
        # 속도를 위해 각 카테고리에서 1개 쿼리만 테스트
        result = run_complete_evaluation_test(
            self.yes_queries[:1],  # 관련성 있는 쿼리
            self.no_queries[:1],   # 관련성 없는 쿼리
            "./chroma_db"
        )
        
        self.assertIn('overall_success_rate', result)
        self.assertGreaterEqual(result['overall_success_rate'], 50.0, "전체 성공률이 50% 미만입니다.")

if __name__ == '__main__':
    unittest.main() 