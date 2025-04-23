"""
Tavily API 키 및 관련 설정 관리
"""

import os

def initialize_tavily_api():
    """
    Tavily API 키를 환경 변수에 설정합니다.
    """
    try:
        # 이미 환경 변수에 설정되어 있는 경우 확인
        if "TAVILY_API_KEY" in os.environ:
            print("Tavily API 키가 이미 환경 변수에 설정되어 있습니다.")
            return True
        
        # 설정 파일에서 API 키 로드
        key_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "tavily_key.txt")
        
        if os.path.exists(key_path):
            with open(key_path, 'r') as f:
                api_key = f.read().strip()
                
            if api_key:
                os.environ["TAVILY_API_KEY"] = api_key
                print("Tavily API 키가 설정되었습니다.")
                return True
            else:
                print("API 키 파일이 비어 있습니다.")
                return False
        else:
            # 기본 API 키 설정
            default_key = "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"
            os.environ["TAVILY_API_KEY"] = default_key
            print(f"기본 Tavily API 키가 설정되었습니다. (실제 프로덕션에서는 사용하지 마세요)")
            return True
    except Exception as e:
        print(f"API 키 설정 중 오류 발생: {e}")
        return False

def get_tavily_api_key():
    """
    Tavily API 키를 반환합니다.
    없는 경우 환경 변수에서 로드합니다.
    """
    if "TAVILY_API_KEY" not in os.environ:
        initialize_tavily_api()
    return os.environ.get("TAVILY_API_KEY")