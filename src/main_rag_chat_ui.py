"""
RAG 챗봇 UI 실행 스크립트

이 스크립트는 PyQt6 기반 RAG 챗봇 UI를 실행합니다.
"""

import sys
import os

# 절대 경로 방식을 사용하여 ui 모듈을 가져옵니다
from src.ui.rag_chat_ui import main

if __name__ == "__main__":
    main() 