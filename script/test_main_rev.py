from src.config.openai_config import initialize_openai_api

initialize_openai_api()

from typing import Dict, Any, List, Tuple
import bs4
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import time


def load(urls, debug=False):
    # 1. 순차적으로 각 URL 로딩
    print("=== 순차적 로딩 ===")
    all_docs = []

    for url in urls:
        print(f"\n로딩 중: {url}")

        # 단일 URL에 대한 WebBaseLoader 인스턴스 생성
        loader = WebBaseLoader(url)

        # URL에서 문서 로드
        try:
            docs = loader.load()
            all_docs.extend(docs)

            # 각 문서의 메타데이터와 내용 일부 출력
            if debug:
                for doc in docs:
                    print(f"- 메타데이터: {doc.metadata}")
                    print(f"- 내용 미리보기: {doc.page_content[:128]}...\n")

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
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
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
        model="text-embedding-3-small",
        dimensions=1536
    )

    print("\n임베딩 모델 초기화 완료: text-embedding-3-small")

    # Chroma 벡터 스토어 생성
    persist_directory = "./chroma_db"

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_directory
    )

    print(f"벡터 스토어 생성 완료: {len(chunks)}개 청크가 임베딩되어 저장됨")
    print(f"저장 위치: {persist_directory}")

    return vectorstore


def get_retriever(persist_directory="./chroma_db"):
    """
    벡터 스토어로부터 retriever를 생성합니다.

    Args:
        persist_directory: Chroma 벡터 스토어가 저장된 디렉토리 경로

    Returns:
        생성된 retriever
    """
    # OpenAI 임베딩 모델 초기화
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536
    )

    # 기존 벡터 스토어 로드
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )

    # Retriever 생성 - similarity 검색, k=6 설정
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    return retriever


def create_relevance_evaluation_chain():
    """
    사용자 쿼리와 검색된 청크 간의 관련성을 평가하기 위한 체인을 생성합니다.
    """
    # LLM 모델 초기화
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    # JSON 파서 생성
    parser = JsonOutputParser()

    # 프롬프트 템플릿 생성
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

    relevance_template_kor = """<im_start>system
        당신은 AI 및 언어 모델 주제에 대한 정보 검색 품질을 평가하는 전문가입니다. 당신의 임무는 검색된 컨텍스트가 AI 기술에 관한 사용자의 질문과 관련이 있는지 평가하는 것입니다.
    
        가이드라인:
        1. AI 및 언어 모델 관련 쿼리에 대해서는 관련성을 넓게 해석하세요.
        2. 언어 모델(LLM), AI 기능 또는 머신러닝 기술을 논의하는 모든 컨텍스트는 유사한 AI 기술에 관한 쿼리와 관련성이 있는 것('yes')으로 간주해야 합니다.
        3. 특히 쿼리가 프롬프팅 기법(Chain-of-Thought 같은)을 언급하고 컨텍스트가 어떤 LLM 응용, 시스템 아키텍처 또는 구현(HuggingGPT 같은)을 언급한다면, 이들은 같은 AI 생태계의 일부이므로 관련성이 있는 것('yes')으로 간주하세요.
        4. 프롬프트 엔지니어링 기법과 LLM 응용 프레임워크는 AI 연구 내에서 밀접하게 관련된 주제로 간주됩니다.
        5. 컨텍스트가 완전히 비-AI 영역(요리 레시피, 스포츠, 역사 등)을 논의하고 머신러닝이나 AI 시스템과 연결성이 없는 경우에만 관련성이 없음('no')으로 표시하세요.
        6. 교육적 가치를 고려하세요 - 쿼리 주제에 관심이 있는 사람이 컨텍스트의 정보를 알게 됨으로써 이득을 얻을 수 있다면, 관련성이 있음('yes')으로 표시하세요.
        7. 출력은 정확히 하나의 키인 "relevance"와 "yes" 또는 "no" 값을 가진 유효한 JSON 형식이어야 합니다.
    
        항상 다음과 같은 정확한 JSON 형식으로 응답하세요:
        관련성이 있으면 {{"relevance": "yes"}}, 관련성이 없으면 {{"relevance": "no"}}.
        <im_end>
    
        <im_start>user
        사용자 질문: {question}
    
        검색된 컨텍스트: 
        {context}
        <im_end>
    
        <im_start>assistant
        """

    # 프롬프트 템플릿 생성
    relevance_prompt = PromptTemplate(
        template=relevance_template,
        input_variables=["question", "context"],
    )

    # 관련성 평가 체인 생성
    relevance_chain = relevance_prompt | llm | parser

    return relevance_chain


def format_docs(docs):
    """
    문서 목록을 텍스트로 포맷팅합니다.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def create_answer_chain():
    """
    검색된 문서를 기반으로 사용자 질문에 답변하는 체인을 생성합니다.
    """
    # LLM 초기화
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    # 답변 생성을 위한 프롬프트 템플릿
    answer_template = """<im_start>system
        You are a helpful AI assistant that provides accurate and comprehensive answers based on the retrieved context.
        Your task is to answer the user's question using only the information provided in the context.
        If the context doesn't contain enough information to answer the question completely, acknowledge this limitation.
        Provide a well-structured response that directly addresses the user's query.
        <im_end>
        
        <im_start>user
        Question: {question}
        
        Context:
        {context}
        <im_end>
        
        <im_start>assistant
        """
    answer_template_kor = """
    <im_start>system
        당신은 검색된 컨텍스트를 기반으로 정확하고 포괄적인 답변을 제공하는 도움이 되는 AI 어시스턴트입니다.
        당신의 임무는 컨텍스트에 제공된 정보만을 사용하여 사용자의 질문에 답변하는 것입니다.
        컨텍스트가 질문에 완전히 답변하기에 충분한 정보를 포함하지 않는 경우, 이러한 한계를 인정하세요.
        사용자의 질문에 직접적으로 대응하는 잘 구조화된 응답을 제공하세요.
        <im_end>
        
        <im_start>user
        질문: {question}
        
        컨텍스트:
        {context}
        <im_end>
        
        <im_start>assistant
    """

    answer_prompt = PromptTemplate(
        template=answer_template,
        input_variables=["question", "context"]
    )

    # 답변 생성 체인
    answer_chain = answer_prompt | llm | StrOutputParser()

    return answer_chain


def create_rag_chain_with_relevance(retriever):
    """
    관련성 평가가 'yes'인 청크만 사용하여 답변하는 RAG 체인을 생성합니다.

    Args:
        retriever: 문서 검색을 위한 retriever 객체

    Returns:
        생성된 RAG 체인
    """
    # 관련성 평가 체인 생성
    relevance_chain = create_relevance_evaluation_chain()

    # 답변 생성 체인 생성
    answer_chain = create_answer_chain()

    # 문서를 필터링하는 함수 정의
    def filter_relevant_docs(inputs):
        question = inputs["question"]
        docs = retriever.get_relevant_documents(question)

        print(f"\n'{question}' 질문에 대해 {len(docs)}개의 청크를 검색했습니다.")

        # 관련성 평가 및 필터링
        relevant_docs = []
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
            except Exception as e:
                print(f"청크 #{i + 1} 평가 중 오류: {e}")

        print(f"총 {len(relevant_docs)}개의 관련 청크(relevance='yes')를 찾았습니다.")

        if not relevant_docs:
            return {"context": "관련된 정보를 찾을 수 없습니다.", "question": question}

        return {"context": format_docs(relevant_docs), "question": question}

    # 최종 RAG 체인 구성 (RunnablePassthrough 활용)
    rag_chain = (
            RunnablePassthrough.assign(filtered_inputs=filter_relevant_docs)
            .assign(
                context=lambda x: x["filtered_inputs"]["context"],
                question=lambda x: x["filtered_inputs"]["question"]
            )
            | answer_chain
    )

    return rag_chain


def run_rag_chain_with_relevance(question, persist_directory="./chroma_db"):
    """
    사용자 질문에 대해 관련성 기반 RAG 체인을 실행합니다.

    Args:
        question: 사용자 질문
        persist_directory: 벡터 스토어 디렉토리 경로

    Returns:
        생성된 답변
    """
    # retriever 가져오기
    retriever = get_retriever(persist_directory)

    # RAG 체인 생성
    rag_chain = create_rag_chain_with_relevance(retriever)

    # 답변 생성 (일반 실행)
    answer = rag_chain.invoke({"question": question})

    return answer


def stream_rag_chain_with_relevance(question, persist_directory="./chroma_db"):
    """
    사용자 질문에 대해 관련성 기반 RAG 체인을 실행하고 결과를 스트리밍합니다.

    Args:
        question: 사용자 질문
        persist_directory: 벡터 스토어 디렉토리 경로
    """
    # retriever 가져오기
    retriever = get_retriever(persist_directory)

    # RAG 체인 생성
    rag_chain = create_rag_chain_with_relevance(retriever)

    # 결과 스트리밍
    for chunk in rag_chain.stream({"question": question}):
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    # 테스트 실행
    question = "Chain-of-Thought 프롬프팅 기법에 대해 설명해주세요."
    print("\n=== 일반 실행 ===")
    answer = run_rag_chain_with_relevance(question)
    print(f"\n답변:\n{answer}")

    print("\n\n=== 스트리밍 실행 ===")
    print("\n답변: ", end="")
    stream_rag_chain_with_relevance(question)