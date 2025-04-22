"""
검색된 문서를 기반으로 사용자 질문에 답변하는 체인을 제공하는 모듈
"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from src.config.openai_config import DEFAULT_MODEL

def create_answer_chain(model: str = DEFAULT_MODEL, temperature: float = 0):
    """
    검색된 문서를 기반으로 사용자 질문에 답변하는 체인을 생성합니다.

    Args:
        model: 사용할 LLM 모델명
        temperature: LLM 모델의 temperature 값

    Returns:
        답변 생성 체인
    """
    # LLM 초기화
    llm = ChatOpenAI(temperature=temperature, model=model)

    # 답변 생성을 위한 프롬프트 템플릿
    answer_template = """<im_start>system
당신은 제공된 컨텍스트를 기반으로 정확하고 유용한 답변을 생성하는 인공지능 비서입니다.
사용자의 질문에 답변할 때는 오직 제공된 컨텍스트에 있는 정보만 사용하세요.
컨텍스트에 충분한 정보가 없는 경우, 그 한계를 솔직하게 인정하세요.
요청에 직접적으로 답변하는 잘 구성된 응답을 제공하세요.
<im_end>

<im_start>user
질문: {question}

컨텍스트:
{context}
<im_end>

<im_start>assistant
"""

    # 프롬프트 템플릿 생성
    answer_prompt = PromptTemplate(
        template=answer_template,
        input_variables=["question", "context"],
    )

    # 답변 생성 체인 생성
    answer_chain = answer_prompt | llm | StrOutputParser()

    return answer_chain 