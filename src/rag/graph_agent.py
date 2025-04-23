"""
LangGraph 기반 RAG 에이전트 모듈

이 모듈은 LangGraph를 이용하여 RAG 에이전트를 구현합니다.
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

from src.config.openai_config import DEFAULT_MODEL, DEFAULT_TEMPERATURE


class GraphState(TypedDict):
    """
    그래프 상태를 나타내는 클래스
    
    Attributes:
        question: 사용자 질문
        generation: LLM이 생성한 답변
        web_search: 웹 검색 여부
        documents: 검색된 문서 목록
    """
    question: str
    generation: str
    web_search: str
    documents: List[Document]


class GraphRAGAgent:
    """LangGraph 기반 RAG 에이전트 클래스"""
    
    def __init__(
        self, 
        retriever, 
        tavily_api_key: str,
        model_name: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE
    ):
        """
        RAG 에이전트 초기화
        
        Args:
            retriever: 문서 검색기
            tavily_api_key: Tavily API 키
            model_name: 사용할 LLM 모델명
            temperature: 생성 다양성 파라미터
        """
        self.retriever = retriever
        self.tavily_api_key = tavily_api_key
        self.model_name = model_name
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
        # 라우터 체인
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at routing a user question to a vectorstore or web search.
            Use the vectorstore for questions on LLM agents, prompt engineering, and adversarial attacks.
            You do not need to be stringent with the keywords in the question related to these topics.
            Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question.
            Return the a JSON with a single key 'datasource' and no premable or explanation."""),
            ("human", "question: {question}"),
        ])
        self.question_router = router_prompt | self.llm | JsonOutputParser()
        
        # 문서 평가 체인
        grader_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a grader assessing relevance of a retrieved document to a user question.
            If the document contains keywords related to the user question, grade it as relevant.
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
            Provide the binary score as a JSON with a single key 'score' and no premable or explanation."""),
            ("human", "question: {question}\n\n document: {document}"),
        ])
        self.retrieval_grader = grader_prompt | self.llm | JsonOutputParser()
        
        # 답변 생성 체인
        generator_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know.
            Use three sentences maximum and keep the answer concise"""),
            ("human", "question: {question}\n\n context: {context}"),
        ])
        self.rag_chain = generator_prompt | self.llm | StrOutputParser()
        
        # 답변 평가 체인
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a grader assessing whether an answer is grounded in / supported by a set of facts.
            Give a binary 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts.
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""),
            ("human", "documents: {documents}\n\n answer: {generation}"),
        ])
        self.answer_grader = answer_prompt | self.llm | JsonOutputParser()

        # 유용성 평가 체인
        usefulness_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a grader assessing whether an answer is useful to resolve a question.
            Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question.
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""),
            ("human", "question: {question}\n\n answer: {generation}"),
        ])
        self.usefulness_grader = usefulness_prompt | self.llm | JsonOutputParser()
    
    def setup_workflow(self):
        """워크플로우 그래프 설정"""
        workflow = StateGraph(GraphState)
        
        # 노드 추가
        workflow.add_node("websearch", self.web_search)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("search_trivily", self.search_trivily)  # Trivily 검색 노드 추가
        
        # 엣지 추가
        workflow.set_conditional_entry_point(
            self.route_question,
            {
                "websearch": "websearch",
                "vectorstore": "retrieve",
            },
        )
        
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "websearch": "websearch",
                "generate": "generate",
                "trivily": "search_trivily",  # Trivily 검색 경로 추가
            },
        )
        workflow.add_edge("websearch", "generate")
        workflow.add_edge("search_trivily", "generate")  # Trivily 검색 후 생성으로 연결
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "websearch",
            },
        )
        
        self.app = workflow.compile()
    
    def retrieve(self, state: GraphState) -> Dict[str, Any]:
        """벡터 스토어에서 문서 검색"""
        print("---RETRIEVE---")
        question = state["question"]
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}
    
    def generate(self, state: GraphState) -> Dict[str, Any]:
        """답변 생성"""
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}
    
    def grade_documents(self, state: GraphState) -> Dict[str, Any]:
        """문서 관련성 평가"""
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = self.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score["score"].lower()
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
        
        return {"documents": filtered_docs, "question": question, "web_search": web_search}
    
    def web_search(self, state: GraphState) -> Dict[str, Any]:
        """웹 검색 수행"""
        print("---WEB SEARCH---")
        question = state["question"]
        documents = state.get("documents", [])
        
        search_results = self.tavily.search(query=question)['results']
        web_results = "\n".join([d["content"] for d in search_results])
        web_doc = Document(page_content=web_results)
        
        if documents:
            documents.append(web_doc)
        else:
            documents = [web_doc]
            
        return {"documents": documents, "question": question}
    
    def search_trivily(self, state: GraphState) -> Dict[str, Any]:
        """
        Trivily 검색 노드: 관련성이 없을 때 추가 검색 수행
        """
        print("---SEARCH TRIVILY---")
        question = state["question"]
        documents = state.get("documents", [])
        
        # Trivily 검색 수행
        search_results = self.tavily.search(
            query=question,
            search_depth="advanced",  # 더 깊은 검색 수행
            include_answer=True,  # 답변 포함
            include_raw_content=True  # 원본 콘텐츠 포함
        )
        
        # 검색 결과 처리
        trivily_docs = []
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
        
        # 기존 문서와 병합
        if documents:
            documents.extend(trivily_docs)
        else:
            documents = trivily_docs
            
        return {"documents": documents, "question": question}
    
    def route_question(self, state: GraphState) -> str:
        """질문 라우팅"""
        print("---ROUTE QUESTION---")
        question = state["question"]
        source = self.question_router.invoke({"question": question})
        return "websearch" if source["datasource"] == "web_search" else "vectorstore"
    
    def decide_to_generate(self, state: GraphState) -> str:
        """답변 생성 여부 결정"""
        print("---ASSESS GRADED DOCUMENTS---")
        web_search = state["web_search"]
        
        if web_search == "Yes":
            # 관련성이 낮은 경우 Trivily 검색 시도
            if len(state.get("documents", [])) < 2:  # 문서가 충분하지 않은 경우
                print("---DECISION: DOCUMENTS NOT RELEVANT, TRY TRIVILY SEARCH---")
                return "trivily"
            print("---DECISION: DOCUMENTS NOT RELEVANT, INCLUDE WEB SEARCH---")
            return "websearch"
        else:
            print("---DECISION: GENERATE---")
            return "generate"
    
    def grade_generation(self, state: GraphState) -> str:
        """생성된 답변 평가"""
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        
        # 할루시네이션 검사
        score = self.answer_grader.invoke({"documents": documents, "generation": generation})
        if score["score"].lower() == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # 유용성 검사
            score = self.usefulness_grader.invoke({"question": question, "generation": generation})
            grade = score["score"].lower()
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
    
    def run(self, question: str) -> Dict[str, Any]:
        """
        RAG 에이전트 실행
        
        Args:
            question: 사용자 질문
            
        Returns:
            Dict: 생성된 답변과 관련 정보 포함
        """
        inputs = {"question": question}
        result = None
        
        for output in self.app.stream(inputs):
            for key, value in output.items():
                print(f"Finished running: {key}")
                result = value
        
        return {
            "question": question,
            "answer": result["generation"],
            "documents": result["documents"],
            "formatted_sources": self._format_sources(result["documents"])
        }
    
    def _format_sources(self, documents: List[Document]) -> str:
        """문서 정보를 포맷팅하여 반환"""
        sources = []
        for i, doc in enumerate(documents, 1):
            metadata = getattr(doc, "metadata", {})
            source = metadata.get("source", "Unknown source")
            content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            sources.append(f"Source {i}: {source}\n{content}\n")
        
        return "\n".join(sources)

def create_graph_rag_agent(retriever, tavily_api_key, model_name=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE):
    """
    LangGraph 기반 RAG 에이전트를 생성합니다.
    
    Args:
        retriever: 문서 검색기
        tavily_api_key: Tavily API 키
        model_name: 사용할 LLM 모델명
        temperature: 생성 다양성 파라미터
    
    Returns:
        GraphRAGAgent: 초기화된 RAG 에이전트
    """
    return GraphRAGAgent(
        retriever=retriever,
        tavily_api_key=tavily_api_key,
        model_name=model_name,
        temperature=temperature
    ) 