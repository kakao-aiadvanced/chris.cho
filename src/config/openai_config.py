"""
OpenAI API 키 및 관련 설정 관리
"""

import os

def initialize_openai_api():
    """
    OpenAI API 키를 환경 변수에 설정합니다.
    """
    try:
        # 이미 환경 변수에 설정되어 있는 경우 확인
        if "OPENAI_API_KEY" in os.environ:
            print("OpenAI API 키가 이미 환경 변수에 설정되어 있습니다.")
            return True
        
        # 설정 파일에서 API 키 로드
        key_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "openai_key.txt")
        
        if os.path.exists(key_path):
            with open(key_path, 'r') as f:
                api_key = f.read().strip()
                
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                print("OpenAI API 키가 설정되었습니다.")
                return True
            else:
                print("API 키 파일이 비어 있습니다.")
                return False
        else:
            print(f"API 키 파일을 찾을 수 없습니다: {key_path}")
            return False
    except Exception as e:
        print(f"API 키 설정 중 오류 발생: {e}")
        return False


# LLM 모델 설정
DEFAULT_MODEL = "gpt-4o-mini"  # 기본 모델
DEFAULT_TEMPERATURE = 0  # 기본 temperature 값
#DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"  # 기본 임베딩 모델
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"  # 기본 임베딩 모델
DEFAULT_EMBEDDING_DIMENSIONS = 1536  # 기본 임베딩 차원
DEFAULT_MAX_PAGE_SIZE = 3