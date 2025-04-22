"""
RAPTOR 인덱스 구축 및 검색 기능을 제공하는 모듈
"""

import os
import json
import numpy as np
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity

from src.embeddings.vector_store import get_embedding_model
from src.text_processing.splitter import split_documents, group_chunks_by_source

def generate_abstract(content: str, llm: ChatOpenAI) -> str:
    """
    주어진 내용에 대한 추상적 요약을 생성합니다.

    Args:
        content: 요약할 내용
        llm: LLM 모델

    Returns:
        생성된 요약
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


def generate_root_abstract(abstracts: List[str], llm: ChatOpenAI) -> str:
    """
    모든 그룹 요약을 통합한 루트 요약을 생성합니다.

    Args:
        abstracts: 통합할 요약 목록
        llm: LLM 모델

    Returns:
        통합된 루트 요약
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


def convert_to_serializable(obj: Any) -> Any:
    """
    객체를 JSON 직렬화 가능한 형태로 변환합니다.

    Args:
        obj: 변환할 객체

    Returns:
        직렬화 가능한 형태로 변환된 객체
    """
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


def create_raptor_index(documents: List, chunk_size: int = 1000, chunk_overlap: int = 200, 
                        persist_directory: str = "./raptor_index", debug: bool = False) -> Dict:
    """
    RAPTOR 인덱스를 생성합니다.

    Args:
        documents: 인덱싱할 문서 목록
        chunk_size: 청크 크기
        chunk_overlap: 청크 간 중복 크기
        persist_directory: 인덱스 저장 디렉토리
        debug: 디버그 정보 출력 여부

    Returns:
        생성된 RAPTOR 인덱스
    """
    # 1. 기본 청크 분할
    chunks = split_documents(documents, chunk_size, chunk_overlap, debug)
    if debug:
        print(f"총 {len(chunks)}개의 청크로 분할되었습니다.")

    # 2. 청크 그룹화 - 소스별
    grouped_chunks = group_chunks_by_source(chunks)
    if debug:
        print(f"총 {len(grouped_chunks)}개의 소스 그룹이 생성되었습니다.")

    # 3. 각 그룹에 대한 요약 생성
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    abstracts = {}

    if debug:
        print("그룹별 요약 생성 중...")
    for group_id, group_chunks in grouped_chunks.items():
        if debug:
            print(f"  소스 '{group_id}' 요약 생성 중...")
        group_content = "\n\n".join([chunk.page_content for chunk in group_chunks[:5]])  # 효율성을 위해 처음 몇 개 청크만 사용
        abstract = generate_abstract(group_content, llm)
        abstracts[group_id] = abstract

    # 4. 루트 요약 생성
    if debug:
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
    if debug:
        print("임베딩 모델 초기화 중...")
    embedding_model = get_embedding_model()

    # 7. 루트 임베딩 생성
    if debug:
        print("루트 임베딩 생성 중...")
    raptor_index["root"]["embedding"] = embedding_model.embed_query(root_abstract)

    # 8. 그룹 노드 생성 및 임베딩
    if debug:
        print("그룹 노드 생성 및 임베딩 중...")
    for group_id, abstract in abstracts.items():
        if debug:
            print(f"  소스 '{group_id}' 처리 중...")
        group_node = {
            "id": group_id,
            "abstract": abstract,
            "embedding": embedding_model.embed_query(abstract),
            "chunks": []
        }

        # 9. 청크 임베딩 생성
        for i, chunk in enumerate(grouped_chunks[group_id]):
            if debug and i % 10 == 0:
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
    if debug:
        print(f"RAPTOR 인덱스를 {persist_directory}에 저장 중...")
    os.makedirs(persist_directory, exist_ok=True)

    # 임베딩을 리스트로 변환 (JSON 직렬화를 위해)
    serializable_index = convert_to_serializable(raptor_index)

    with open(os.path.join(persist_directory, "raptor_index.json"), "w") as f:
        json.dump(serializable_index, f)

    if debug:
        print("RAPTOR 인덱스 생성 완료!")
    return raptor_index


def load_raptor_index(persist_directory: str = "./raptor_index") -> Dict:
    """
    저장된 RAPTOR 인덱스를 로드합니다.

    Args:
        persist_directory: 인덱스가 저장된 디렉토리

    Returns:
        로드된 RAPTOR 인덱스
    """
    index_path = os.path.join(persist_directory, "raptor_index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"RAPTOR 인덱스 파일을 찾을 수 없습니다: {index_path}")

    with open(index_path, "r") as f:
        raptor_index = json.load(f)

    print(f"RAPTOR 인덱스를 {persist_directory}에서 로드했습니다.")
    return raptor_index


def get_raptor_retriever(persist_directory: str = "./raptor_index"):
    """
    RAPTOR 인덱스를 사용하는 retriever를 생성합니다.

    Args:
        persist_directory: 인덱스가 저장된 디렉토리

    Returns:
        RAPTOR retriever
    """
    # RAPTOR 인덱스 로드
    raptor_index = load_raptor_index(persist_directory)
    
    # 임베딩 모델 초기화
    embedding_model = get_embedding_model()
    
    def retrieve_func(query: str, top_k: int = 6):
        """
        쿼리에 기반하여 관련 문서를 검색하는 함수

        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수

        Returns:
            검색된 문서 목록
        """
        return raptor_retrieval(query, raptor_index, embedding_model, top_k)
    
    class RaptorRetriever:
        def get_relevant_documents(self, query: str, top_k: int = 6):
            return retrieve_func(query, top_k)
    
    return RaptorRetriever()


def raptor_retrieval(question: str, raptor_index: Dict, embedding_model = None, top_k: int = 6):
    """
    RAPTOR 인덱스를 사용하여 관련 문서를 검색합니다.

    Args:
        question: 검색 질문
        raptor_index: RAPTOR 인덱스
        embedding_model: 임베딩 모델 (없으면 기본값 사용)
        top_k: 반환할 문서 수

    Returns:
        검색된 문서 목록
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()
    
    # 질문을 임베딩
    query_embedding = embedding_model.embed_query(question)
    
    # 1단계: 루트 노드 탐색
    root_node = raptor_index["root"]
    
    # 2단계: 관련 그룹 노드 찾기
    groups = root_node["children"]
    group_similarities = []
    
    for group in groups:
        group_embedding = np.array(group["embedding"])
        similarity = cosine_similarity([query_embedding], [group_embedding])[0][0]
        group_similarities.append((group, similarity))
    
    # 유사도에 따라 그룹 정렬
    sorted_groups = sorted(group_similarities, key=lambda x: x[1], reverse=True)
    
    # 상위 그룹 선택 (상위 3개)
    top_groups = sorted_groups[:3]
    
    # 3단계: 상위 그룹 내에서 관련 청크 찾기
    all_chunk_similarities = []
    
    for group, group_similarity in top_groups:
        chunks = group["chunks"]
        
        for chunk in chunks:
            chunk_embedding = np.array(chunk["embedding"])
            similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
            
            # 그룹 유사도를 고려한 최종 점수 계산
            # 그룹 유사도와 청크 유사도의 가중 평균
            combined_score = 0.3 * group_similarity + 0.7 * similarity
            
            all_chunk_similarities.append((chunk, combined_score))
    
    # 유사도에 따라 청크 정렬
    sorted_chunks = sorted(all_chunk_similarities, key=lambda x: x[1], reverse=True)
    
    # 상위 top_k개 청크 선택
    top_chunks = sorted_chunks[:top_k]
    
    # 결과 문서 생성
    from langchain_core.documents import Document
    
    results = []
    for chunk, score in top_chunks:
        doc = Document(
            page_content=chunk["content"],
            metadata={
                "source": chunk["metadata"]["source"],
                "score": score,
                "chunk_id": chunk["id"]
            }
        )
        results.append(doc)
    
    return results 