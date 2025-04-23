"""
LangGraph 기반 향상된 RAG 에이전트 모듈

이 모듈은 LangGraph를 이용하여 향상된 RAG 에이전트를 구현합니다.
각 노드와 엣지를 그래프로 구성하여 상태 기반 RAG 시스템을 구축합니다.
"""

import os
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from tavily import TavilyClient

from src.config import openai_config
from src.utils.smart_cache_util import smart_cache, KeyStrategy, initialize_cache_with_smart_cache

class GraphState(TypedDict):
    """
    워크플로우의 상태를 나타냅니다.

    Attributes:
        question: 사용자 질문
        documents: 검색된 문서 목록
        answer: 생성된 답변
        relevance_count: 관련성 체크 재귀 카운트
        hallucination_count: 유해성 체크 생성 카운트
        sources: 출처 정보
        is_relevant: 관련성 여부
        is_hallucination: 유해성 여부
    """
    question: str
    documents: List[Document]
    answer: str
    relevance_count: int
    hallucination_count: int
    sources: List[dict]
    is_relevant: bool
    is_hallucination: bool
    final_answer: str


class EnhancedGraphRAGAgent:
    """LangGraph 기반 향상된 RAG 에이전트 클래스"""
    
    def __init__(
        self, 
        retriever, 
        tavily_api_key: str,
        model_name: str = openai_config.DEFAULT_MODEL,
        temperature: float = openai_config.DEFAULT_TEMPERATURE
    ):
        """
        RAG 에이전트 초기화
        
        Args:
            retriever: 문서 검색기
            tavily_api_key: Tavily API 키
            model_name: 사용할 LLM 모델명
            temperature: 생성 다양성 파라미터
        """
        initialize_cache_with_smart_cache(1)
        self.retriever = retriever
        self.tavily_api_key = tavily_api_key
        self.model_name = model_name
        openai_config.DEFAULT_MODEL = model_name
        openai_config.DEFAULT_TEMPERATURE = temperature
        self.temperature = temperature
        
        self.setup_apis()
        self.setup_chains()
        self.setup_workflow()
    
    def setup_apis(self):
        """API 클라이언트 설정"""
        self.tavily = TavilyClient(api_key=self.tavily_api_key)
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature
        )
    
    def setup_chains(self):
        """프롬프트 체인 설정"""
        # 문서 평가 체인
        relevance_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a grader assessing relevance of a retrieved document to a user question.
            If the document contains keywords related to the user question, grade it as relevant.
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
            Provide the binary score as a JSON with a single key 'score' and no premable or explanation."""),
            ("human", "question: {question}\n\n document: {document}"),
        ])
        self.retrieval_grader = relevance_prompt | self.llm | JsonOutputParser()
        
        # 답변 생성 체인
        generator_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know.
            Use three sentences maximum and keep the answer concise"""),
            ("human", "question: {question}\n\n context: {context}"),
        ])
        self.rag_chain = generator_prompt | self.llm | StrOutputParser()
        
        # 할루시네이션 평가 체인
        hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a grader assessing whether an answer is grounded in / supported by a set of facts.
            Give a binary 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts.
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""),
            ("human", "documents: {documents}\n\n answer: {answer}"),
        ])
        self.hallucination_grader = hallucination_prompt | self.llm | JsonOutputParser()
    
    def setup_workflow(self):
        """워크플로우 그래프 설정"""
        workflow = StateGraph(GraphState)
        
        # 노드 추가
        workflow.add_node("docs_retrieval", self.docs_retrieval)
        workflow.add_node("relevance_checker", self.relevance_checker)
        workflow.add_node("search_trivily", self.search_trivily)
        workflow.add_node("generate_answer", self.generate_answer)
        workflow.add_node("hallucination_checker", self.hallucination_checker)
        #workflow.add_node("regenerate_answer", self.regenerate_answer)
        workflow.add_node("finalize_answer", self.finalize_answer)
        workflow.add_node("handle_relevance_failure", self.handle_relevance_failure)
        workflow.add_node("handle_hallucination_failure", self.handle_hallucination_failure)
        
        # 진입점 설정
        workflow.set_entry_point("docs_retrieval")
        
        # 기본 엣지 설정
        workflow.add_edge("docs_retrieval", "relevance_checker")
        workflow.add_edge("search_trivily", "relevance_checker")
        workflow.add_edge("generate_answer", "hallucination_checker")
        #workflow.add_edge("regenerate_answer", "hallucination_checker")
        workflow.add_edge("finalize_answer", END)
        workflow.add_edge("handle_relevance_failure", END)
        workflow.add_edge("handle_hallucination_failure", END)
        
        # 조건부 엣지 설정
        workflow.add_conditional_edges(
            "relevance_checker",
            self.check_relevance_result,
            {
                "relevant": "generate_answer",
                "not_relevant": "search_trivily",
                "failed": "handle_relevance_failure"
            }
        )
        
        workflow.add_conditional_edges(
            "hallucination_checker",
            self.check_hallucination_result,
            {
                "safe": "finalize_answer",
                "unsafe": "generate_answer",
                "failed": "handle_hallucination_failure"
            }
        )
        
        self.app = workflow.compile()
    
    def docs_retrieval(self, state: GraphState) -> Dict[str, Any]:
        """
        문서 검색 노드: 쿼리를 받아 검색 기능을 통해 관련 문서 검색
        """
        print("---DOCUMENT RETRIEVAL---")
        query = state["question"]
        
        # 벡터 스토어에서 문서 검색
        documents = self.retriever.invoke(query)
        
        # 출처 정보 추출 (URL, 제목 등)
        sources = []
        for doc in documents:
            if hasattr(doc, 'metadata'):
                sources.append({
                    'title': doc.metadata.get('title', 'Unknown'),
                    'url': doc.metadata.get('source', 'Unknown URL')
                })
        
        return {
            "question": query, 
            "documents": documents,
            "sources": sources,
            "relevance_count": 0,
            "hallucination_count": 0,
            "answer": ""
        }
    
    def relevance_checker(self, state: GraphState, debug=False) -> Dict[str, Any]:
        """
        관련성 체크 노드: 검색된 문서가 쿼리와 관련 있는지 확인
        """
        print("---RELEVANCE CHECKER---")
        query = state["question"]
        documents = state["documents"]
        relevance_count = state["relevance_count"]
        sources = state["sources"]
        
        # 관련성 체크 로직
        relevant_docs = []
        is_relevant = False
        
        for doc in documents:
            if debug:
                print(f"[debug] query={query}")
                print(f"[debug] document={doc.page_content}")

            score = self.retrieval_grader.invoke(
                {"question": query, "document": doc.page_content}
            )
            grade = score["score"].lower()
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                relevant_docs.append(doc)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
        
        # 관련 문서가 하나 이상 있으면 관련성 있음
        is_relevant = len(relevant_docs) > 0
        
        return {
            "question": query,
            "documents": relevant_docs,
            "sources": sources,
            "relevance_count": relevance_count,
            "hallucination_count": 0,
            "answer": "",
            "is_relevant": is_relevant
        }

    @smart_cache(prefix='search_trivily', key_strategy=KeyStrategy.CONTENT)
    def search_trivily(self, state: GraphState) -> Dict[str, Any]:
        """
        Trivily 검색 노드: 관련성이 없을 때 추가 검색 수행
        """
        print("---SEARCH TRIVILY---")
        query = state["question"]
        relevance_count = state["relevance_count"] + 1  # 재귀 카운트 증가
        
        # Trivily 검색 수행
        search_results = self.tavily.search(
            query=query,
            search_depth="advanced",  # 더 깊은 검색 수행
            include_answer=True,  # 답변 포함
            include_raw_content=True  # 원본 콘텐츠 포함
        )
        
        # 검색 결과 처리
        trivily_docs = []
        sources = []
        if search_results.get('results'):
            for result in search_results['results']:
                content = result.get('content', '')
                if result.get('raw_content'):
                    content += f"\n\nRaw content: {result['raw_content']}"
                trivily_docs.append(Document(
                    page_content=content,
                    metadata={
                        'source': result.get('url', 'Unknown'),
                        'title': result.get('title', 'No title'),
                        'score': result.get('score', 0.0)
                    }
                ))
                sources.append({
                    'title': result.get('title', 'No title'),
                    'url': result.get('url', 'Unknown URL')
                })
        
        return {
            "question": query,
            "documents": trivily_docs,
            "sources": sources,
            "relevance_count": relevance_count,
            "hallucination_count": 0,
            "answer": ""
        }
    
    def generate_answer(self, state: GraphState) -> Dict[str, Any]:
        """
        답변 생성 노드: 관련 문서를 기반으로 답변 생성
        """
        print("---GENERATE ANSWER---")
        question = state["question"]
        documents = state["documents"]
        sources = state["sources"]
        
        # 문서 내용 추출
        docs_content = []
        for doc in documents:
            docs_content.append(doc.page_content)
        context = "\n\n".join(docs_content)
        
        # 답변 생성
        answer = self.rag_chain.invoke({"context": context, "question": question})
        
        return {
            "question": question,
            "documents": documents,
            "answer": answer,
            "sources": sources,
            "hallucination_count": 0
        }
    
    def hallucination_checker(self, state: GraphState) -> Dict[str, Any]:
        """
        유해성 체크 노드: 생성된 답변의 유해성 확인
        """
        print("---HALLUCINATION CHECKER---")
        query = state["question"]
        documents = state["documents"]
        answer = state["answer"]
        hallucination_count = state["hallucination_count"] + 1
        sources = state["sources"]
        
        # 문서 내용 추출
        docs_content = [doc.page_content for doc in documents]
        docs_text = "\n\n".join(docs_content)
        
        # 할루시네이션 검사
        score = self.hallucination_grader.invoke({"documents": docs_text, "answer": answer})
        is_hallucination = score["score"].lower() != "yes"
        
        if is_hallucination:
            print("---DECISION: ANSWER HAS HALLUCINATION---")
        else:
            print("---DECISION: ANSWER IS GROUNDED IN DOCUMENTS---")
        
        return {
            "question": query,
            "documents": documents,
            "answer": answer,
            "sources": sources,
            "hallucination_count": hallucination_count,
            "is_hallucination": is_hallucination
        }
    '''
    def regenerate_answer(self, state: GraphState) -> Dict[str, Any]:
        """
        답변 재생성 노드: 유해한 답변이 감지되어 답변 재생성
        """
        print("---REGENERATE ANSWER---")
        query = state["question"]
        documents = state["documents"]
        hallucination_count = state["hallucination_count"] + 1  # 재생성 카운트 증가
        sources = state["sources"]
        
        # 문서 내용 추출
        docs_content = [doc.page_content for doc in documents]
        context = "\n\n".join(docs_content)
        
        # 안전한 답변 생성을 위한 프롬프트 추가
        safe_generator_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            IMPORTANT: Only include information that is directly supported by the provided context.
            Do not add any information that is not in the context.
            If you don't know the answer, just say that you don't know.
            Use three sentences maximum and keep the answer concise."""),
            ("human", "question: {question}\n\n context: {context}"),
        ])
        safe_rag_chain = safe_generator_prompt | self.llm | StrOutputParser()
        
        # 안전한 답변 재생성
        new_answer = safe_rag_chain.invoke({"context": context, "question": query})
        
        return {
            "question": query,
            "documents": documents,
            "answer": new_answer,
            "sources": sources,
            "hallucination_count": hallucination_count
        }
        '''

    def finalize_answer(self, state: GraphState) -> Dict[str, Any]:
        """
        최종 답변 생성 노드: 출처 정보 추가하여 답변 완성
        """
        print("---FINALIZE ANSWER---")
        answer = state["answer"]
        sources = state["sources"]

        # 출처 정보 추가
        source_text = "\n\n출처:"
        for i, source in enumerate(sources, 1):
            source_text += f"\n{i}. {source['title']} - {source['url']}"

        final_answer = answer + source_text
        state['final_answer'] = final_answer
        #print(f"최종 답변: {final_answer}")

        # 상태에 final_answer 추가 (딕셔너리 반환 아님)
        return {"final_answer": final_answer}

    def handle_relevance_failure(self, state: GraphState) -> Dict[str, Any]:
        """
        관련성 실패 처리 노드: 최대 재귀 회수 초과 시 실패 메시지 출력
        """
        print("---RELEVANCE FAILURE---")
        return {
            "final_answer": "failed: not relevant"
        }
    
    def handle_hallucination_failure(self, state: GraphState) -> Dict[str, Any]:
        """
        유해성 실패 처리 노드: 최대 재생성 회수 초과 시 실패 메시지 출력
        """
        print("---HALLUCINATION FAILURE---")
        return {
            "final_answer": "failed: hallucination"
        }
    
    def check_relevance_result(self, state: GraphState) -> str:
        """
        관련성 체크 결과 확인
        """
        is_relevant = state.get("is_relevant", False)
        relevance_count = state.get("relevance_count", 0)
        
        if is_relevant:
            return "relevant"
        elif relevance_count >= 1:  # 최대 1회 재귀
            return "failed"
        else:
            return "not_relevant"
    
    def check_hallucination_result(self, state: GraphState) -> str:
        """
        유해성 체크 결과 확인
        """
        is_hallucination = state.get("is_hallucination", False)
        hallucination_count = state.get("hallucination_count", 0)
        
        if not is_hallucination:
            return "safe"
        elif hallucination_count >= 1+1:  # 최대 1회 재생성
            return "failed"
        else:
            return "unsafe"
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        RAG 에이전트 실행
        
        Args:
            query: 사용자 질문
            
        Returns:
            Dict: 생성된 답변과 관련 정보 포함
        """
        inputs = {"question": query}
        result = None

        result = self.app.invoke(inputs)

        if result and "final_answer" in result:
            return {
                "question": query,
                "answer": result["final_answer"]
            }
        else:
            print("디버깅:", result)  # 실제 결과 구조 확인
            return {
                "question": query,
                "answer": "답변을 생성하는 데 실패했습니다."
            }


def create_enhanced_graph_rag_agent(retriever, tavily_api_key, model_name=openai_config.DEFAULT_MODEL, temperature=openai_config.DEFAULT_TEMPERATURE):
    """
    향상된 LangGraph 기반 RAG 에이전트를 생성합니다.
    
    Args:
        retriever: 문서 검색기
        tavily_api_key: Tavily API 키
        model_name: 사용할 LLM 모델명
        temperature: 생성 다양성 파라미터
    
    Returns:
        EnhancedGraphRAGAgent: 초기화된 RAG 에이전트
    """
    return EnhancedGraphRAGAgent(
        retriever=retriever,
        tavily_api_key=tavily_api_key,
        model_name=model_name,
        temperature=temperature
    ) 