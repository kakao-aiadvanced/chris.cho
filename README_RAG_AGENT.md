# LangGraph 기반 RAG 에이전트

이 프로젝트는 LangGraph를 활용하여 상태 기반 RAG(Retrieval-Augmented Generation) 에이전트를 구현합니다. 기본 버전과 향상된 버전을 제공하여 다양한 활용 사례에 맞게 사용할 수 있습니다.

## 기능

### 기본 버전 (GraphRAGAgent)
- 벡터 스토어와 웹 검색을 통합한 RAG 시스템
- LangGraph를 사용한 상태 기반 워크플로우
- 문서 관련성 평가 및 할루시네이션 검사
- 문서 출처 추적 및 포맷팅

### 향상된 버전 (EnhancedGraphRAGAgent)
- 다중 검색 소스 지원 (벡터 스토어, 웹 검색, 지식 그래프 등)
- 문서 클러스터링 및 우선순위 지정
- 사용자 정의 프롬프트 템플릿
- 답변 신뢰도 평가
- 상세한 결과 분석 및 메타데이터 제공

## 설치 방법

1. 기본 의존성 설치
```bash
pip install -r requirements.txt
```

2. 테스트 및 추가 기능을 위한 의존성 설치
```bash
pip install -r requirements_test.txt
```

## 사용 방법

### 기본 버전

```python
from src.config.tavily_config import get_tavily_api_key
from src.rag.graph_agent import create_graph_rag_agent
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 벡터 스토어 로드
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
)

# RAG 에이전트 생성
agent = create_graph_rag_agent(
    retriever=vectorstore.as_retriever(),
    tavily_api_key=get_tavily_api_key()
)

# 쿼리 실행
result = agent.run("프롬프트 엔지니어링이란 무엇인가요?")

# 결과 출력
print(f"질문: {result['question']}")
print(f"답변: {result['answer']}")
print(f"출처: {result['formatted_sources']}")
```

### 향상된 버전

```python
from src.config.tavily_config import get_tavily_api_key
from src.rag.enhanced_graph_agent import (
    create_enhanced_graph_rag_agent,
    create_default_search_sources,
    EnhancedPromptTemplates
)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 벡터 스토어 로드
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
)

# 검색 소스 생성
search_sources = create_default_search_sources(vectorstore.as_retriever())

# 사용자 정의 프롬프트 템플릿 (선택 사항)
custom_prompts = EnhancedPromptTemplates(
    answer_generator="""당신은 질문에 답변하는 전문가입니다.
    다음 컨텍스트를 사용하여 질문에 답변하세요.
    모르는 경우 모른다고 답변하세요.
    최대 세 문장으로 간결하게 답변하세요."""
)

# 향상된 RAG 에이전트 생성
agent = create_enhanced_graph_rag_agent(
    search_sources=search_sources,
    tavily_api_key=get_tavily_api_key(),
    prompt_templates=custom_prompts,
    enable_clustering=True,
    enable_confidence=True
)

# 쿼리 실행
result = agent.run("프롬프트 엔지니어링이란 무엇인가요?")

# 결과 출력
print(f"질문: {result['question']}")
print(f"답변: {result['answer']}")
print(f"신뢰도: {result['confidence']}")
print(f"출처: {result['formatted_sources']}")
print(f"분석: {result['analysis']}")
```

## 메인 모듈에서 사용하기

메인 모듈에서는 `run_graph_rag_query` 함수를 사용하여 간편하게 실행할 수 있습니다:

```python
from src.main import run_graph_rag_query

# 기존 RAG 체인으로 실행
result = run_graph_rag_query("프롬프트 엔지니어링이란 무엇인가요?")

print(f"질문: {result['question']}")
print(f"답변: {result['answer']}")
print(f"출처: {result['formatted_sources']}")
```

## 테스트 실행

테스트를 실행하려면 다음 명령을 사용하세요:

```bash
# 모든 단위 테스트 실행
python src/tests/run_tests.py

# 통합 테스트 실행 (실제 API 호출)
python src/tests/run_tests.py --int

# 특정 테스트 실행
python src/tests/run_tests.py test_graph_agent.TestGraphRAGAgent.test_retrieve
```

## 확장 방법

1. 새로운 검색 소스 추가:
```python
from src.rag.enhanced_graph_agent import SearchSource

# 새로운 검색 소스 추가
knowledge_graph_retriever = ... # 지식 그래프 검색기 구현
new_source = SearchSource(
    name="knowledge_graph",
    retriever=knowledge_graph_retriever,
    weight=0.9,
    enabled=True
)

# 기존 소스와 함께 사용
search_sources = create_default_search_sources(vectorstore.as_retriever())
search_sources.append(new_source)

# 에이전트 생성
agent = create_enhanced_graph_rag_agent(search_sources=search_sources, ...)
```

2. 사용자 정의 프롬프트 템플릿:
```python
from src.rag.enhanced_graph_agent import EnhancedPromptTemplates

custom_prompts = EnhancedPromptTemplates(
    router="당신은 질문을 적절한 데이터 소스로 라우팅하는 전문가입니다. 사용 가능한 소스: {datasources}",
    answer_generator="당신은 한국어로 답변하는 도우미입니다. 다음 컨텍스트를 사용하여 최대 세 문장으로 답변하세요."
)

agent = create_enhanced_graph_rag_agent(prompt_templates=custom_prompts, ...)
```

## 주의사항

- Tavily API 키가 필요합니다. `src/config/tavily_key.txt` 파일에 저장하거나 환경 변수로 설정하세요.
- 기본 버전과 향상된 버전 모두 LangGraph를 사용하며, 이는 상태 관리를 위한 추가 요구 사항이 있습니다.
- 향상된 버전의 클러스터링은 scikit-learn을 사용하므로 테스트 의존성을 설치해야 합니다.
- 신뢰도 평가와 분석 기능은 추가 API 호출을 발생시킬 수 있습니다. 