# LangGraph 기반 향상된 RAG 에이전트

이 프로젝트는 LangGraph를 활용하여 상태 기반 RAG(Retrieval-Augmented Generation) 에이전트를 구현합니다. 특히 할루시네이션 탐지와 관련성 검사를 통합한 향상된 검색 증강 생성 시스템을 제공합니다.

## 주요 기능

- **그래프 기반 상태 관리**: LangGraph를 활용하여 복잡한 워크플로우를 그래프로 구성
- **관련성 평가**: 검색된 문서의 관련성을 평가하고 필터링
- **할루시네이션 탐지**: 생성된 답변이 검색된 문서에 기반하는지 확인
- **다중 검색 경로**: 로컬 벡터 스토어와 Tavily 웹 검색을 통합
- **적응형 검색 전략**: 관련 문서를 찾지 못하면 대체 검색 경로로 전환
- **출처 추적**: 생성된 답변에 대한 출처 정보 제공

## 아키텍처

EnhancedGraphRAGAgent는 다음과 같은 노드로 구성된 그래프 기반 워크플로우를 구현합니다:

1. **docs_retrieval**: 벡터 스토어에서 문서 검색
2. **relevance_checker**: 검색된 문서의 관련성 평가
3. **search_trivily**: 벡터 스토어에서 관련 문서를 찾지 못한 경우 웹 검색 실행
4. **generate_answer**: 관련 문서를 기반으로 답변 생성
5. **hallucination_checker**: 생성된 답변이 문서에 근거하는지 검사
6. **finalize_answer**: 출처 정보를 포함하여 최종 답변 구성
7. **handle_relevance_failure** / **handle_hallucination_failure**: 검색/생성 실패 처리

## 워크플로우 다이어그램

```
[docs_retrieval] → [relevance_checker] → (관련성 있음) → [generate_answer] → [hallucination_checker] → (근거 있음) → [finalize_answer]
                                       → (관련성 없음) → [search_trivily] → [relevance_checker]
                                       → (실패) → [handle_relevance_failure]
                                                                         → (근거 없음) → [generate_answer]
                                                                         → (실패) → [handle_hallucination_failure]
```

## 설치 방법

1. 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

2. API 키 설정:
   - OpenAI API 키: `config/openai_key.txt` 파일에 저장
   - Tavily API 키: `config/tavily_key.txt` 파일에 저장

## 사용 방법

### 기본 사용법

```python
from src.config.tavily_config import get_tavily_api_key
from src.retrieval.retriever import get_retriever
from src.rag.enhanced_graph_agent import EnhancedGraphRAGAgent

# 검색기 생성
retriever = get_retriever()

# RAG 에이전트 생성
agent = EnhancedGraphRAGAgent(
    retriever=retriever,
    tavily_api_key=get_tavily_api_key()
)

# 질문 실행
result = agent.run("프롬프트 엔지니어링이란 무엇인가요?")

# 결과 출력
print(f"질문: {result['question']}")
print(f"답변: {result['answer']}")
```

### 메인 모듈에서 사용하기

```python
from src.main_enhanced_graph_rag_agent import run_graph_rag_query

# 질문 실행
result = run_graph_rag_query("프롬프트 엔지니어링이란 무엇인가요?")

# 결과 출력
print(f"질문: {result['question']}")
print(f"답변: {result['answer']}")
```

## 주요 구현 클래스 및 함수

### GraphState
워크플로우의 상태를 관리하는 TypedDict 클래스입니다:

```python
class GraphState(TypedDict):
    question: str                # 사용자 질문
    documents: List[Document]    # 검색된 문서 목록
    answer: str                  # 생성된 답변
    relevance_count: int         # 관련성 체크 재귀 카운트
    hallucination_count: int     # 유해성 체크 생성 카운트
    sources: List[dict]          # 출처 정보
    is_relevant: bool            # 관련성 여부
    is_hallucination: bool       # 유해성 여부
```

### EnhancedGraphRAGAgent
LangGraph 기반 향상된 RAG 에이전트 클래스입니다:

```python
class EnhancedGraphRAGAgent:
    def __init__(self, retriever, tavily_api_key, model_name=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE):
        # 에이전트 초기화
        
    def setup_apis(self):
        # API 클라이언트 설정
        
    def setup_chains(self):
        # 프롬프트 체인 설정
        
    def setup_workflow(self):
        # 워크플로우 그래프 설정
        
    def run(self, query: str) -> Dict[str, Any]:
        # RAG 에이전트 실행
```

## 프롬프트 템플릿

에이전트는 다음과 같은 주요 프롬프트 템플릿을 사용합니다:

1. **문서 관련성 평가**: 검색된 문서가 질문과 관련이 있는지 평가
2. **답변 생성**: 관련 문서를 기반으로 답변을 생성
3. **할루시네이션 평가**: 생성된 답변이 문서에 근거하는지 평가

## 유의사항

- 관련성 체크와 할루시네이션 검사는 최대 1회씩 재시도됩니다.
- 문서 검색에 실패하면 Tavily 웹 검색으로 대체됩니다.
- 모든 검색에 실패하면 적절한 오류 메시지가 반환됩니다.
- 할루시네이션이 감지되면 답변을 재생성하여 문서에 더 충실한 답변을 생성합니다.

## 확장 방향

- **다중 검색 소스**: 추가 벡터 스토어나 검색 엔진 통합
- **사용자 정의 프롬프트**: 도메인별 프롬프트 템플릿 구성
- **계층적 검색**: RAPTOR와 같은 계층적 검색 방식 통합
- **다국어 지원**: 다양한 언어에 대한 프롬프트 최적화

## 라이센스

이 프로젝트는 MIT 라이센스 하에 공개되어 있습니다. 