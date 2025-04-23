"""
RAG 챗봇 UI 실행 스크립트

이 스크립트는 PyQt6 기반 RAG 챗봇 UI를 실행합니다.
"""

import sys
import os

# 현재 파일의 위치를 확인하고 상위 디렉토리를 path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # src 디렉토리를 경로에 추가

# 직접적인 상대 경로 사용
from ui.rag_chat_ui import main

if __name__ == "__main__":
    main() 