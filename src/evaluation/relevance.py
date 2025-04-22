"""
검색 결과와 쿼리 간의 관련성을 평가하는 모듈
"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from src.config.openai_config import DEFAULT_MODEL

def create_relevance_evaluation_chain(model: str = DEFAULT_MODEL, temperature: float = 0):
    """
    관련성 평가 체인을 생성합니다.

    Args:
        model: 사용할 LLM 모델명
        temperature: LLM 모델의 temperature 값

    Returns:
        관련성 평가 체인
    """
    llm = ChatOpenAI(temperature=temperature, model=model)
    parser = JsonOutputParser()

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

    relevance_prompt = PromptTemplate(
        template=relevance_template,
        input_variables=["question", "context"],
    )

    relevance_chain = relevance_prompt | llm | parser

    return relevance_chain