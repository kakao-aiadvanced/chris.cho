"""
생성된 답변에 할루시네이션이 있는지 평가하는 모듈
"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from src.config.openai_config import DEFAULT_MODEL

def create_hallucination_evaluation_chain(model: str = DEFAULT_MODEL, temperature: float = 0):
    """
    할루시네이션 평가 체인을 생성합니다.

    Args:
        model: 사용할 LLM 모델명
        temperature: LLM 모델의 temperature 값

    Returns:
        할루시네이션 평가 체인
    """
    llm = ChatOpenAI(temperature=temperature, model=model)
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


def create_enhanced_hallucination_evaluation_chain(model: str = DEFAULT_MODEL, temperature: float = 0):
    """
    개선된 할루시네이션 평가 체인을 생성합니다.
    이 체인은 단순히 환각 여부뿐만 아니라, 환각이 발생한 부분과 이유도 함께 제공합니다.

    Args:
        model: 사용할 LLM 모델명
        temperature: LLM 모델의 temperature 값

    Returns:
        개선된 할루시네이션 평가 체인
    """
    llm = ChatOpenAI(temperature=temperature, model=model)
    parser = JsonOutputParser()

    enhanced_hallucination_template = """<im_start>system
                                        당신은 AI 응답에서 환각(hallucination)을 분석하는 전문가입니다. 
                                        환각이란 제공된 컨텍스트에 없는 내용을 AI가 지어내거나 잘못 해석하여 생성한 정보를 의미합니다.
                                        
                                        당신의 임무는 주어진 사용자 질문, 검색된 컨텍스트, 그리고 생성된 답변을 세밀하게 분석하여 
                                        답변에 환각이 포함되어 있는지 평가하고, 환각이 발견된 경우 해당 부분과 이유를 명확히 설명하는 것입니다.
                                        
                                        평가 가이드라인:
                                        1. 답변의 모든 사실적 주장이 제공된 컨텍스트에 명시적으로 포함되어 있는지 확인하세요.
                                        2. 컨텍스트에서 직접적으로 추론할 수 없는 추가 정보나 세부 사항이 있는지 확인하세요.
                                        3. 컨텍스트의 정보를 왜곡하거나 잘못 해석한 내용이 있는지 확인하세요.
                                        4. 답변이 컨텍스트의 범위를 벗어난 주제로 확장되었는지 확인하세요.
                                        5. 답변에 잘못된 날짜, 이름, 통계 또는 숫자가 포함되어 있는지 확인하세요.
                                        6. 일반적인 상식이나 널리 알려진 정보(예: "지구는 둥글다")의 언급은 환각으로 간주하지 마세요.
                                        7. 답변이 컨텍스트의 정보를 재구성하거나 요약한 경우, 정보의 의미가 보존되었는지 확인하세요.
                                        
                                        출력 형식(JSON):
                                        {{
                                          "hallucination": "yes" 또는 "no",
                                          "analysis": 환각 여부 평가에 대한 간결한 설명,
                                          "problematic_parts": 환각이 발견된 경우 답변에서 문제가 된 부분들의 배열(발견되지 않았으면 빈 배열),
                                          "suggestions": 답변을 개선하기 위한 제안사항 배열(환각이 없는 경우에도 더 좋은 답변을 위한 제안)
                                        }}
                                        
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
    enhanced_hallucination_prompt = PromptTemplate(
        template=enhanced_hallucination_template,
        input_variables=["question", "context", "answer"],
    )

    enhanced_hallucination_chain = enhanced_hallucination_prompt | llm | parser

    return enhanced_hallucination_chain