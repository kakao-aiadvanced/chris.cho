from config.open_api_config import initialize_openai_api

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
    """
    URL 목록에서 문서를 로드합니다.

    Args:
        urls: 로드할 URL 목록
        debug: 디버그 정보 출력 여부

    Returns:
        로드된 문서 목록
    """
    print("=== 순차적 로딩 ===")
    all_docs = []

    for url in urls:
        print(f"\n로딩 중: {url}")
        loader = WebBaseLoader(url)

        try:
            docs = loader.load()
            all_docs.extend(docs)

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
    문서를 청크로 분할합니다.

    Args:
        documents: 분할할 문서 목록
        chunk_size: 청크 크기
        chunk_overlap: 청크 간 중복 크기

    Returns:
        분할된 청크 목록
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"원본 문서 수: {len(documents)}")
    print(f"분할 후 청크 수: {len(chunks)}")

    print("\n== 처음 3개 청크 개요 ==")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n청크 #{i + 1}")
        print(f"소스: {chunk.metadata.get('source', '출처 없음')}")
        print(f"문자 수: {len(chunk.page_content)}")
        print(f"내용 미리보기: {chunk.page_content[:100]}...")

    return chunks


def create_vector_store(chunks):
    """
    청크로부터 벡터 스토어를 생성합니다.

    Args:
        chunks: 벡터화할 청크 목록

    Returns:
        생성된 벡터 스토어
    """
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536
    )

    print("\n임베딩 모델 초기화 완료: text-embedding-3-small")

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
        persist_directory: 벡터 스토어 디렉토리

    Returns:
        생성된 retriever
    """
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536
    )

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    return retriever


def create_relevance_evaluation_chain():
    """
    관련성 평가 체인을 생성합니다.

    Returns:
        관련성 평가 체인
    """
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    parser = JsonOutputParser()

    relevance_template = """<im_start>system
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

    relevance_prompt = PromptTemplate(
        template=relevance_template,
        input_variables=["question", "context"],
    )

    relevance_chain = relevance_prompt | llm | parser

    return relevance_chain


def create_hallucination_evaluation_chain():
    """
    생성된 답변에 할루시네이션이 있는지 평가하는 체인을 생성합니다.

    Returns:
        할루시네이션 평가 체인
    """
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    parser = JsonOutputParser()

    hallucination_template = """<im_start>system
당신은 AI 응답에서 환각(hallucination)을 탐지하는 전문가입니다. 환각이란 제공된 컨텍스트에 없는 내용을 AI가 지어내거나 잘못 해석하여 생성한 정보를 의미합니다.

당신의 임무는 주어진 사용자 질문, 검색된 컨텍스트, 그리고 생성된 답변을 분석하여 답변에 환각이 포함되어 있는지 평가하는 것입니다.

평가 가이드라인:
1. 답변의 모든 사실적 주장이 제공된 컨텍스트에 명시적으로 포함되어 있는지 확인하세요.
2. 컨텍스트에서 직접적으로 추론할 수 없는 추가 정보나 세부 사항이 있는지 확인하세요.
3. 컨텍스트의 정보를 왜곡하거나 잘못 해석한 내용이 있는지 확인하세요.
4. 답변이 컨텍스트의 범위를 벗어난 주제로 확장되었는지 확인하세요.
5. 답변에 잘못된 날짜, 이름, 통계 또는 숫자가 포함되어 있는지 확인하세요.
6. 일반적인 상식이나 널리 알려진 정보(예: "지구는 둥글다")의 언급은 환각으로 간주하지 마세요.
7. 답변이 컨텍스트의 정보를 재구성하거나 요약한 경우, 정보의 의미가 보존되었는지 확인하세요.

출력 형식:
- 환각이 발견된 경우: {{"hallucination": "yes"}}
- 환각이 발견되지 않은 경우: {{"hallucination": "no"}}

답변의 형식이나 구조가 아닌 사실적 정확성에 초점을 맞추세요.
<im_end>

<im_start>user
사용자 질문: {question}

검색된 컨텍스트:
{context}

생성된 답변:
{answer}
<im_end>

<im_start>assistant
"""

    hallucination_prompt = PromptTemplate(
        template=hallucination_template,
        input_variables=["question", "context", "answer"],
    )

    hallucination_chain = hallucination_prompt | llm | parser

    return hallucination_chain


def create_enhanced_hallucination_evaluation_chain():
    """
    개선된 할루시네이션 평가 체인을 생성합니다.
    """
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    parser = JsonOutputParser()

    hallucination_template = """<im_start>system
당신은 AI 응답에서 환각(hallucination)을 탐지하는 전문가입니다. 환각이란 제공된 컨텍스트에 없는 내용을 AI가 지어내거나 잘못 해석하여 생성한 정보를 의미합니다.

당신의 임무는 주어진 사용자 질문, 검색된 컨텍스트, 그리고 생성된 답변을 분석하여 답변에 환각이 포함되어 있는지 평가하는 것입니다.

AI 모델과 기술 관련 정확한 정보:
- 2025년 4월 현재 OpenAI의 가장 최신 모델은 GPT-4.0, GPT-4 Turbo와 GPT-4o 시리즈입니다.
- GPT-5, GPT-6, GPT-7 등은 아직 출시되지 않았으며, 이에 대한 구체적인 성능이나 특징을 서술하는 것은 환각입니다.
- 현존하는 주요 LLM 모델: Claude 3(Anthropic), Gemini(Google), Llama 3(Meta), Mixtral(Mistral AI)
- 새로운 AI 기술이나 프레임워크가 언급될 경우, 검색된 컨텍스트에서 명확하게 언급되어 있어야 합니다.

평가 가이드라인:
1. 답변의 모든 사실적 주장이 제공된 컨텍스트에 명시적으로 포함되어 있는지 엄격하게 확인하세요.
2. 컨텍스트에서 직접적으로 추론할 수 없는 추가 정보나 세부 사항이 있는지 확인하세요.
3. 컨텍스트의 정보를 왜곡하거나 잘못 해석한 내용이 있는지 확인하세요.
4. 답변이 컨텍스트의 범위를 벗어난 주제로 확장되었는지 확인하세요.
5. 답변에 잘못된 날짜, 이름, 통계, 숫자, 모델 버전이 포함되어 있는지 확인하세요.
6. 일반적인 상식이나 널리 알려진 정보(예: "지구는 둥글다")의 언급은 환각으로 간주하지 마세요.
7. 답변이 컨텍스트의 정보를 재구성하거나 요약한 경우, 정보의 의미가 보존되었는지 확인하세요.
8. 존재하지 않는 AI 모델이나 기술에 대한 구체적인 설명이 있다면 반드시 환각으로 표시하세요.
9. 컨텍스트에 명시되지 않은 연구 논문, 발표, 출시 정보에 대한 언급은 환각으로 간주하세요.

중요: 사용자가 특정 모델이나 기술에 대해 질문했다고 해서, 그 모델이나 기술이 실제로 존재한다고 가정하지 마세요. 컨텍스트에 해당 정보가 있는지 확인하세요.

출력 형식:
- 환각이 발견된 경우: {{"hallucination": "yes"}}
- 환각이 발견되지 않은 경우: {{"hallucination": "no"}}

답변의 형식이나 구조가 아닌 사실적 정확성에 초점을 맞추세요.
<im_end>

<im_start>user
사용자 질문: {question}

검색된 컨텍스트:
{context}

생성된 답변:
{answer}
<im_end>

<im_start>assistant
"""

    hallucination_prompt = PromptTemplate(
        template=hallucination_template,
        input_variables=["question", "context", "answer"],
    )

    hallucination_chain = hallucination_prompt | llm | parser

    return hallucination_chain


def format_docs(docs):
    """
    문서 목록을 텍스트로 포맷팅합니다.

    Args:
        docs: 포맷팅할 문서 목록

    Returns:
        포맷팅된 텍스트
    """
    return "\n\n".join(doc.page_content for doc in docs)


def create_answer_chain():
    """
    답변 생성 체인을 생성합니다.

    Returns:
        답변 생성 체인
    """
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    answer_template = """<im_start>system
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

    answer_chain = answer_prompt | llm | StrOutputParser()

    return answer_chain


def create_rag_chain_with_hallucination_check(retriever):
    """
    할루시네이션 검사 및 재시도 기능이 포함된 RAG 체인을 생성합니다.

    Args:
        retriever: 문서 검색을 위한 retriever

    Returns:
        생성된 RAG 체인
    """
    # 필요한 체인들 생성
    relevance_chain = create_relevance_evaluation_chain()
    answer_chain = create_answer_chain()
    hallucination_chain = create_enhanced_hallucination_evaluation_chain()

    def generate_answer_with_hallucination_check(inputs, max_retries=1, current_retry=0):
        """
        할루시네이션 검사 및 재귀적 재시도를 통해 답변을 생성하는 재귀 함수

        Args:
            inputs: 입력 데이터
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
                "has_hallucination": has_hallucination,
                "retried": current_retry
            }

        except Exception as e:
            print(f"할루시네이션 검사 중 오류 발생: {e}")
            # 오류 발생 시 현재 답변 반환
            return {
                "answer": answer,
                "sources": sources,
                "has_hallucination": False,
                "retried": current_retry
            }

    # 최종 RAG 체인 구성
    chain = RunnablePassthrough.assign(
        result=lambda x: generate_answer_with_hallucination_check(x, max_retries=1)
    ).assign(
        answer=lambda x: x["result"]["answer"],
        sources=lambda x: x["result"]["sources"],
        has_hallucination=lambda x: x["result"]["has_hallucination"],
        retried=lambda x: x["result"]["retried"]
    )

    return chain


def format_sources(sources):
    """
    출처 정보를 텍스트로 포맷팅합니다.

    Args:
        sources: 포맷팅할 출처 정보

    Returns:
        포맷팅된 출처 텍스트
    """
    if not sources:
        return "출처 정보 없음"

    formatted_text = "\n\n참고 출처:\n"
    for idx, source in enumerate(sources, 1):
        formatted_text += f"{idx}. {source['source']}\n"

    return formatted_text


def run_rag_chain_with_hallucination_check(question, persist_directory="./chroma_db"):
    """
    할루시네이션 검사가 포함된 RAG 체인을 실행합니다.

    Args:
        question: 사용자 질문
        persist_directory: 벡터 스토어 디렉토리

    Returns:
        생성된 답변 및 메타데이터
    """
    # retriever 가져오기
    retriever = get_retriever(persist_directory)

    # 할루시네이션 검사가 포함된 RAG 체인 생성
    rag_chain = create_rag_chain_with_hallucination_check(retriever)

    # 체인 실행
    result = rag_chain.invoke({"question": question})

    # 답변 및 메타데이터 추출
    answer = result["answer"]
    sources = result["sources"]
    has_hallucination = result["has_hallucination"]
    retried = result["retried"]

    # 답변 포맷팅
    formatted_answer = f"[***] AI의 답변:{answer}\n\n{format_sources(sources)}"

    if has_hallucination:
        formatted_answer += "\n\n(주의: 이 답변에는 할루시네이션이 포함되어 있을 수 있습니다.)"

    if retried > 0:
        formatted_answer += f"\n\n(참고: 할루시네이션 감지로 {retried}회 재생성 수행함)"

    return formatted_answer


def stream_rag_chain_with_hallucination_check(question, persist_directory="./chroma_db"):
    """
    할루시네이션 검사가 포함된 RAG 체인을 실행하고 결과를 스트리밍합니다.

    Args:
        question: 사용자 질문
        persist_directory: 벡터 스토어 디렉토리
    """
    # 일반 실행 결과를 스트리밍 방식으로 출력
    formatted_answer = run_rag_chain_with_hallucination_check(question, persist_directory)

    for chunk in formatted_answer:
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    # 테스트 실행
    #good_question = "Chain-of-Thought 프롬프팅 기법에 대해 설명해주세요."
    hall_question = "GPT-7 모델의 성능과 특징에 대해 자세히 설명해주세요."
    question = hall_question
    print("\n=== 할루시네이션 검사 포함 실행 ===")
    answer = run_rag_chain_with_hallucination_check(question)
    print(f"\n답변:\n{answer}")

    print("\n\n=== 스트리밍 실행 ===")
    print("\n답변: ", end="")
    stream_rag_chain_with_hallucination_check(question)