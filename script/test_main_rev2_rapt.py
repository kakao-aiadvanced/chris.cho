from src.config.openai_config import initialize_openai_api
from sklearn.metrics.pairwise import cosine_similarity

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
import os
import json
import numpy as np

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
    문서를 청크로 분할합니다. (기존 코드 유지)
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


def group_chunks_by_source(chunks):
    """
    청크를 소스별로 그룹화합니다.
    """
    groups = {}
    for chunk in chunks:
        source = chunk.metadata.get('source', 'unknown')
        if source not in groups:
            groups[source] = []
        groups[source].append(chunk)
    return groups


def generate_abstract(content, llm):
    """
    주어진 내용에 대한 추상적 요약을 생성합니다.
    """
    abstract_prompt = PromptTemplate(
        template="""<im_start>system
당신은 텍스트 내용을 간결하고 정확하게 요약하는 전문가입니다.
주어진 내용의 핵심 주제와 중요 개념을 포함하는 200단어 이내의 요약을 생성하세요.
<im_end>

<im_start>user
다음 텍스트의 주요 내용을 요약해주세요:

{content}
<im_end>

<im_start>assistant
""",
        input_variables=["content"]
    )

    chain = abstract_prompt | llm | StrOutputParser()
    return chain.invoke({"content": content[:10000]})  # 내용이 너무 길면 잘라냄


def generate_root_abstract(abstracts, llm):
    """
    모든 그룹 요약을 통합한 루트 요약을 생성합니다.
    """
    all_abstracts = "\n\n".join(abstracts)
    root_prompt = PromptTemplate(
        template="""<im_start>system
여러 문서 요약본들을 통합하여 전체 문서 컬렉션의 핵심 주제와 범위를 나타내는 통합 요약을 생성하세요.
<im_end>

<im_start>user
다음은 여러 문서의 요약입니다. 이들을 통합한 전체적인 요약을 150단어 이내로 작성해주세요:

{abstracts}
<im_end>

<im_start>assistant
""",
        input_variables=["abstracts"]
    )

    chain = root_prompt | llm | StrOutputParser()
    return chain.invoke({"abstracts": all_abstracts})


def create_raptor_index(documents, chunk_size=1000, chunk_overlap=200, persist_directory="./raptor_index"):
    """
    RAPTOR 인덱스를 생성합니다.
    """
    # 1. 기본 청크 분할
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    print(f"총 {len(chunks)}개의 청크로 분할되었습니다.")

    # 2. 청크 그룹화 - 소스별
    grouped_chunks = group_chunks_by_source(chunks)
    print(f"총 {len(grouped_chunks)}개의 소스 그룹이 생성되었습니다.")

    # 3. 각 그룹에 대한 요약 생성
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    abstracts = {}

    print("그룹별 요약 생성 중...")
    for group_id, group_chunks in grouped_chunks.items():
        print(f"  소스 '{group_id}' 요약 생성 중...")
        group_content = "\n\n".join([chunk.page_content for chunk in group_chunks[:5]])  # 효율성을 위해 처음 몇 개 청크만 사용
        abstract = generate_abstract(group_content, llm)
        abstracts[group_id] = abstract

    # 4. 루트 요약 생성
    print("루트 요약 생성 중...")
    root_abstract = generate_root_abstract(list(abstracts.values()), llm)

    # 5. 계층적 인덱스 구조 생성
    raptor_index = {
        "root": {
            "abstract": root_abstract,
            "children": [],
            "embedding": None
        }
    }

    # 6. 임베딩 모델 초기화
    print("임베딩 모델 초기화 중...")
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536
    )

    # 7. 루트 임베딩 생성
    print("루트 임베딩 생성 중...")
    raptor_index["root"]["embedding"] = embedding_model.embed_query(root_abstract)

    # 8. 그룹 노드 생성 및 임베딩
    print("그룹 노드 생성 및 임베딩 중...")
    for group_id, abstract in abstracts.items():
        print(f"  소스 '{group_id}' 처리 중...")
        group_node = {
            "id": group_id,
            "abstract": abstract,
            "embedding": embedding_model.embed_query(abstract),
            "chunks": []
        }

        # 9. 청크 임베딩 생성
        for i, chunk in enumerate(grouped_chunks[group_id]):
            if i % 10 == 0:
                print(f"    청크 {i}/{len(grouped_chunks[group_id])} 임베딩 생성 중...")

            embedding = embedding_model.embed_query(chunk.page_content)
            chunk_data = {
                "id": f"{group_id}_chunk_{i}",
                "content": chunk.page_content,
                "metadata": chunk.metadata,
                "embedding": embedding
            }
            group_node["chunks"].append(chunk_data)

        raptor_index["root"]["children"].append(group_node)

    # 10. 인덱스 저장
    print(f"RAPTOR 인덱스를 {persist_directory}에 저장 중...")
    os.makedirs(persist_directory, exist_ok=True)

    # 임베딩을 리스트로 변환 (JSON 직렬화를 위해)
    serializable_index = convert_to_serializable(raptor_index)

    with open(os.path.join(persist_directory, "raptor_index.json"), "w") as f:
        json.dump(serializable_index, f)

    print("RAPTOR 인덱스 생성 완료!")
    return raptor_index


def convert_to_serializable(obj):
    """객체를 JSON 직렬화 가능한 형태로 변환합니다."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            result[k] = convert_to_serializable(v)
        return result
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def load_raptor_index(persist_directory="./raptor_index"):
    """
    저장된 RAPTOR 인덱스를 로드합니다.
    """
    index_path = os.path.join(persist_directory, "raptor_index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"RAPTOR 인덱스 파일을 찾을 수 없습니다: {index_path}")

    with open(index_path, "r") as f:
        raptor_index = json.load(f)

    print(f"RAPTOR 인덱스를 {persist_directory}에서 로드했습니다.")
    return raptor_index


def get_raptor_retriever(persist_directory="./raptor_index"):
    """
    RAPTOR 검색기를 생성합니다.
    """
    raptor_index = load_raptor_index(persist_directory)

    def retriever_func(query, top_k=6):
        return raptor_retrieval(query, raptor_index, top_k=top_k)

    # 고급 검색 클래스로 래핑할 수도 있지만, 간단한 함수로 구현
    class RaptorRetriever:
        def get_relevant_documents(self, query, top_k=6):
            return retriever_func(query, top_k)

    return RaptorRetriever()


def raptor_retrieval(question, raptor_index, top_k=6):
    """
    RAPTOR 검색을 수행합니다.
    """
    print(f"\n'{question}' 질문에 대해 RAPTOR 검색 수행 중...")

    # 1. 임베딩 모델 초기화
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536
    )

    # 2. 질문 임베딩
    query_embedding = embedding_model.embed_query(question)

    # 3. 그룹 수준에서 유사성 계산
    group_similarities = []
    for child in raptor_index["root"]["children"]:
        # 임베딩이 리스트로 저장되어 있을 경우 다시 numpy 배열로 변환
        child_embedding = np.array(child["embedding"])
        query_np = np.array(query_embedding)

        # 벡터 유사도 계산
        sim = cosine_similarity([query_np], [child_embedding])[0][0]
        group_similarities.append((child, sim))

    # 4. 상위 그룹 선택 (top_k의 2배 정도로 충분히 많은 그룹 선택)
    group_count = min(len(group_similarities), max(2, top_k // 3))
    top_groups = sorted(group_similarities, key=lambda x: x[1], reverse=True)[:group_count]

    print(f"총 {len(group_similarities)}개 그룹 중 상위 {len(top_groups)}개 그룹 선택됨")

    # 5. 선택된 그룹 내에서 청크 수준 검색
    all_candidate_chunks = []
    for group, group_sim in top_groups:
        print(f"  그룹 '{group['id']}' (유사도: {group_sim:.4f})에서 청크 검색 중...")

        for chunk in group["chunks"]:
            chunk_embedding = np.array(chunk["embedding"])
            query_np = np.array(query_embedding)

            # 청크 유사도 계산
            sim = cosine_similarity([query_np], [chunk_embedding])[0][0]

            # 그룹 유사도와 청크 유사도를 결합한 최종 점수 계산
            # 그룹 유사도에 가중치 0.2, 청크 유사도에 가중치 0.8 적용
            combined_score = 0.2 * group_sim + 0.8 * sim

            all_candidate_chunks.append((chunk, combined_score))

    # 6. 최종 상위 청크 선택
    top_chunks = sorted(all_candidate_chunks, key=lambda x: x[1], reverse=True)[:top_k]

    # 7. LangChain Document 형식으로 변환하여 반환
    from langchain_core.documents import Document

    result_docs = []
    for chunk, score in top_chunks:
        doc = Document(
            page_content=chunk["content"],
            metadata={
                **chunk["metadata"],
                "score": score,
                "raptor_chunk_id": chunk["id"]
            }
        )
        result_docs.append(doc)

    print(f"총 {len(result_docs)}개의 관련 청크를 반환합니다.")
    return result_docs

from test_main_rev2 import (create_relevance_evaluation_chain,
                            create_answer_chain,
                            create_enhanced_hallucination_evaluation_chain,
                            format_docs,
                            format_sources)
def create_raptor_rag_chain_with_hallucination_check(retriever):
    """
    RAPTOR 검색기를 사용하는 RAG 체인을 생성합니다.
    """
    # 필요한 체인들 생성 (코드 재사용)
    relevance_chain = create_relevance_evaluation_chain()
    answer_chain = create_answer_chain()
    hallucination_chain = create_enhanced_hallucination_evaluation_chain()

    def generate_answer_with_raptor(inputs, max_retries=1, current_retry=0):
        """
        RAPTOR 검색과 할루시네이션 검사를 통해 답변을 생성합니다.
        """
        question = inputs["question"]
        docs = retriever.get_relevant_documents(question)  # RAPTOR 검색기 사용

        print(f"\n'{question}' 질문에 대해 {len(docs)}개의 청크를 검색했습니다.")

        # 이하 기존 코드와 동일 (관련성 평가, 답변 생성, 할루시네이션 검사)
        relevant_docs = []
        sources = []

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

        formatted_context = format_docs(relevant_docs)

        answer = answer_chain.invoke({
            "question": question,
            "context": formatted_context
        })

        try:
            hallucination_result = hallucination_chain.invoke({
                "question": question,
                "context": formatted_context,
                "answer": answer
            })

            has_hallucination = hallucination_result.get("hallucination") == "yes"
            print(f"할루시네이션 검사 결과: {'있음' if has_hallucination else '없음'}")

            if has_hallucination and current_retry < max_retries:
                print(f"할루시네이션 감지. 재시도 중... ({current_retry + 1}/{max_retries})")
                return generate_answer_with_raptor(
                    inputs,
                    max_retries=max_retries,
                    current_retry=current_retry + 1
                )

            return {
                "answer": answer,
                "sources": sources,
                "has_hallucination": has_hallucination,
                "retried": current_retry
            }

        except Exception as e:
            print(f"할루시네이션 검사 중 오류 발생: {e}")
            return {
                "answer": answer,
                "sources": sources,
                "has_hallucination": False,
                "retried": current_retry
            }

    # 최종 RAG 체인 구성
    chain = RunnablePassthrough.assign(
        result=lambda x: generate_answer_with_raptor(x, max_retries=1)
    ).assign(
        answer=lambda x: x["result"]["answer"],
        sources=lambda x: x["result"]["sources"],
        has_hallucination=lambda x: x["result"]["has_hallucination"],
        retried=lambda x: x["result"]["retried"]
    )

    return chain


def run_raptor_rag_chain(question, persist_directory="./raptor_index"):
    """
    RAPTOR 검색기를 사용하는 RAG 체인을 실행합니다.
    """
    # RAPTOR 검색기 가져오기
    retriever = get_raptor_retriever(persist_directory)

    # RAPTOR 검색을 사용하는 RAG 체인 생성
    rag_chain = create_raptor_rag_chain_with_hallucination_check(retriever)

    # 체인 실행
    result = rag_chain.invoke({"question": question})

    # 답변 및 메타데이터 추출
    answer = result["answer"]
    sources = result["sources"]
    has_hallucination = result["has_hallucination"]
    retried = result["retried"]

    # 답변 포맷팅 (기존과 동일)
    formatted_answer = f"[***] AI의 답변:{answer}\n\n{format_sources(sources)}"

    if has_hallucination:
        formatted_answer += "\n\n(주의: 이 답변에는 할루시네이션이 포함되어 있을 수 있습니다.)"

    if retried > 0:
        formatted_answer += f"\n\n(참고: 할루시네이션 감지로 {retried}회 재생성 수행함)"

    return formatted_answer

if __name__ == "__main__":
    # 1. 문서 로드 (기존과 동일)
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    documents = load(urls)

    # 2. RAPTOR 인덱스 생성 (새로운 방식)
    raptor_index = create_raptor_index(documents)

    # 3. 검색 및 질문 처리 (RAPTOR 사용)
    question = "Chain-of-Thought 프롬프팅 기법에 대해 설명해주세요."
    answer = run_raptor_rag_chain(question)
    print(f"\n답변:\n{answer}")

    # 할루시네이션 테스트
    hall_question = "GPT-7 모델의 성능과 특징에 대해 자세히 설명해주세요."
    hall_answer = run_raptor_rag_chain(hall_question)
    print(f"\n할루시네이션 테스트 답변:\n{hall_answer}")