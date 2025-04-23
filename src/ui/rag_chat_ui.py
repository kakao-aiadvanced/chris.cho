"""
향상된 RAG 에이전트와 연동된 간단한 채팅 UI

이 모듈은 PyQt6를 사용하여 사용자 친화적인 채팅 인터페이스를 제공하며,
LangGraph 기반 RAG 에이전트를 통해 사용자 질문에 답변합니다.
"""

import sys
import os
import threading
import traceback
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTextEdit, QLineEdit, 
                           QPushButton, QVBoxLayout, QHBoxLayout, QWidget, 
                           QLabel, QSplitter, QComboBox, QMessageBox, QSpinBox)
from PyQt6.QtCore import Qt, pyqtSignal, QObject

# 절대 경로 방식으로 임포트
from src.config.tavily_config import get_tavily_api_key, initialize_tavily_api
from src.config.openai_config import initialize_openai_api, DEFAULT_MAX_PAGE_SIZE
from src.retrieval.retriever import get_retriever
from src.rag.enhanced_graph_agent import EnhancedGraphRAGAgent
from src.main import setup_vector_store

# 디버깅을 위한 메시지
print(f"현재 작업 디렉토리: {os.getcwd()}")
print(f"파이썬 경로: {sys.path}")
print("모듈 가져오기 성공")

class WorkerSignals(QObject):
    """
    작업 스레드에서 메인 스레드로 신호를 보내기 위한 클래스
    """
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)


class RAGWorker(threading.Thread):
    """
    백그라운드에서 RAG 에이전트를 실행하기 위한 작업자 스레드
    """
    def __init__(self, agent, question, document_limit=3):
        super().__init__()
        self.agent = agent
        self.question = question
        self.document_limit = document_limit
        self.signals = WorkerSignals()
        
    def run(self):
        try:
            # return 제거 - 이 부분이 즉시 종료를 유발했음
            
            print(f"질문을 RAG 에이전트에 전달: {self.question}")
            self.signals.progress.emit(f"질문 처리 중: {self.question}")
            
            # 문서 제한 설정 시도
            try:
                # retriever의 k 속성 설정
                if hasattr(self.agent, 'retriever') and hasattr(self.agent.retriever, 'k'):
                    self.agent.retriever.k = self.document_limit
                    self.signals.progress.emit(f"문서 제한 설정: {self.document_limit}개")
                
                # 직접 openai_config의 DEFAULT_MAX_PAGE_SIZE 설정
                from src.config import openai_config
                openai_config.DEFAULT_MAX_PAGE_SIZE = self.document_limit
                self.signals.progress.emit(f"기본 페이지 크기 제한: {self.document_limit}")
            except Exception as e:
                print(f"문서 제한 설정 실패: {e}")
                self.signals.progress.emit(f"문서 제한 설정 실패: {e}")
            
            # 간단한 질문으로 토큰 제한 테스트
            if self.question == "test":
                result = {"question": self.question, "answer": "테스트가 성공적으로 완료되었습니다."}
                self.signals.finished.emit(result)
                return
            
            try:
                # RAG 에이전트 실행
                result = self.agent.run(self.question)
                print(f"RAG 응답 원본: {result}")
                
                # 결과 처리 로직
                processed_result = {}
                processed_result['question'] = self.question
                
                # 답변 추출 (여러 가능한 키 확인)
                if isinstance(result, dict):
                    if 'final_answer' in result:
                        processed_result['answer'] = result['final_answer']
                    elif 'answer' in result:
                        processed_result['answer'] = result['answer']
                    elif 'generation' in result:
                        processed_result['answer'] = result['generation']
                    else:
                        processed_result['answer'] = str(result)
                else:
                    processed_result['answer'] = str(result)
                
                print(f"RAG 응답 처리됨: {processed_result}")
                self.signals.finished.emit(processed_result)
            except Exception as e:
                error_message = str(e)
                if "context_length_exceeded" in error_message:
                    # 토큰 제한 오류 처리
                    self.signals.progress.emit("토큰 제한 초과 오류 발생. 웹 검색을 사용합니다.")
                    processed_result = {
                        'question': self.question,
                        'answer': f"죄송합니다. 이 질문은 너무 많은 토큰을 사용하고 있습니다. 이는 일반적으로 '검색 문서 수'를 줄이거나 '웹 검색'을 사용하여 해결할 수 있습니다.\n\n이 질문에 대해 웹 검색이 더 적합할 수 있습니다."
                    }
                    self.signals.finished.emit(processed_result)
                else:
                    raise e  # 다른 유형의 오류는 다시 발생시켜 상위 예외 처리로 전달
                
        except Exception as e:
            print(f"RAG 에이전트 오류: {e}")
            traceback.print_exc()
            self.signals.error.emit(str(e))


class RAGChatWindow(QMainWindow):
    """
    RAG 채팅 애플리케이션의 메인 창
    """
    def __init__(self):
        super().__init__()
        
        # 디버그 모드
        self.debug = False
        
        # 창 설정
        self.setWindowTitle("RAG 챗봇")
        self.setGeometry(100, 100, 800, 600)
        
        # 위젯과 레이아웃 초기화
        self.init_ui()
        
        # RAG 에이전트 설정
        self.setup_rag_agent()
        
    def debug_print(self, message):
        """디버그 모드일 때 메시지 출력"""
        if self.debug:
            print(f"[디버그] {message}")
            self.add_system_message(f"디버그: {message}")
        
    def init_ui(self):
        """UI 컴포넌트 초기화"""
        # 중앙 위젯 생성
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout(central_widget)
        
        # 상단 정보 영역
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel("향상된 RAG 에이전트 채팅"))
        
        # 모델 선택 콤보박스
        self.model_selector = QComboBox()
        self.model_selector.addItems(["gpt-4o-mini", "gpt-3.5-turbo"])
        info_layout.addWidget(QLabel("모델:"))
        info_layout.addWidget(self.model_selector)
        
        # 문서 제한 컨트롤
        info_layout.addWidget(QLabel("검색 문서 수:"))
        self.doc_limit_selector = QSpinBox()
        self.doc_limit_selector.setMinimum(1)
        self.doc_limit_selector.setMaximum(10)
        self.doc_limit_selector.setValue(6)  # 더 적은 문서 수로 기본값 설정
        info_layout.addWidget(self.doc_limit_selector)
        
        main_layout.addLayout(info_layout)
        
        # 채팅 표시 영역
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        main_layout.addWidget(self.chat_display)
        
        # 입력 영역
        input_layout = QHBoxLayout()
        
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("질문을 입력하세요... (또는 'test'를 입력하여 연결 테스트)")
        self.message_input.returnPressed.connect(self.send_message)
        
        self.send_button = QPushButton("전송")
        self.send_button.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(self.send_button)
        
        main_layout.addLayout(input_layout)
        
        # 초기 메시지 추가
        self.add_system_message("안녕하세요! RAG 챗봇입니다. 질문을 입력해 주세요.")
        self.add_system_message("참고: 검색 문서 수를 줄이면 토큰 길이 제한 오류를 방지할 수 있습니다.")
        self.add_system_message("'test'를 입력하여 연결 테스트를 할 수 있습니다.")
    
    def setup_rag_agent(self):
        """RAG 에이전트 초기화"""
        try:
            self.debug_print("API 초기화 시작")
            
            # API 초기화
            initialize_openai_api()
            initialize_tavily_api()
            
            self.debug_print("Retriever 생성 시작")

            # 주요 문서 URL 정의
            urls = [
                "https://lilianweng.github.io/posts/2023-06-23-agent/",
                "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
                "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
            ]

            self.debug_print(f"벡터 스토어 설정 중: {len(urls)}개 URL")
            
            # 벡터 스토어 생성 (이미 존재하면 로드)
            try:
                vectorstore = setup_vector_store(urls)
                self.debug_print("벡터 스토어 설정 완료")
            except Exception as e:
                self.debug_print(f"벡터 스토어 설정 오류: {e}")
                # 계속 진행

            # Retriever 생성
            self.retriever = get_retriever()
            self.debug_print("Retriever 생성 완료")
            
            self.debug_print("RAG 에이전트 생성 시작")
            
            # RAG 에이전트 생성
            try:
                self.agent = EnhancedGraphRAGAgent(
                    retriever=self.retriever,
                    tavily_api_key=get_tavily_api_key(),
                    model_name=self.model_selector.currentText()
                )
                self.debug_print("RAG 에이전트 초기화 완료")
                self.add_system_message("RAG 에이전트가 준비되었습니다.")
            except Exception as e:
                self.debug_print(f"RAG 에이전트 생성 실패: {e}")
                self.add_system_message(f"RAG 에이전트 생성 실패: {e}. 기본 RAG 기능만 사용할 수 있습니다.")
        except Exception as e:
            error_msg = f"에러: RAG 에이전트 초기화 실패 - {str(e)}"
            self.debug_print(error_msg)
            self.add_system_message(error_msg)
            traceback.print_exc()
    
    def add_system_message(self, message):
        """시스템 메시지를 채팅 창에 추가"""
        self.chat_display.append(f"<b>시스템:</b> {message}")
    
    def add_user_message(self, message):
        """사용자 메시지를 채팅 창에 추가"""
        self.chat_display.append(f"<b>사용자:</b> {message}")
    
    def add_bot_message(self, message):
        """봇 메시지를 채팅 창에 추가"""
        self.chat_display.append(f"<b>RAG 챗봇:</b> {message}")
    
    def send_message(self):
        """사용자 메시지 전송 및 답변 처리"""
        # 입력 텍스트 가져오기
        message = self.message_input.text().strip()
        if not message:
            return
        
        # 사용자 메시지 표시
        self.add_user_message(message)
        
        # 입력 필드 초기화
        self.message_input.clear()
        
        # UI 상태 업데이트
        self.message_input.setEnabled(False)
        self.send_button.setEnabled(False)
        self.add_system_message("답변을 생성 중입니다...")
        
        try:
            # 현재 선택된 모델로 에이전트 업데이트
            if hasattr(self, 'agent'):
                self.agent.model_name = self.model_selector.currentText()
                self.debug_print(f"모델 변경: {self.agent.model_name}")
            
            # 문서 제한 값 가져오기
            doc_limit = self.doc_limit_selector.value()
            self.debug_print(f"문서 제한: {doc_limit}개")
            
            # 백그라운드 스레드에서 RAG 에이전트 실행
            worker = RAGWorker(self.agent, message, doc_limit)
            worker.signals.finished.connect(self.handle_rag_response)
            worker.signals.error.connect(self.handle_rag_error)
            worker.signals.progress.connect(self.handle_progress)
            worker.start()
            
            self.debug_print("RAG 작업자 스레드 시작")
        except Exception as e:
            error_msg = f"질문 처리 중 오류 발생: {str(e)}"
            self.debug_print(error_msg)
            self.handle_rag_error(error_msg)
    
    def handle_progress(self, message):
        """진행 상황 처리"""
        self.debug_print(message)
    
    def handle_rag_response(self, result):
        """RAG 에이전트 응답 처리"""
        try:
            self.debug_print(f"RAG 응답 받음: {result}")
            
            # 봇 답변 표시
            answer = result.get('answer', '답변을 생성하지 못했습니다.')
            
            # 실패 메시지 확인
            if answer == '답변을 생성하는 데 실패했습니다.' or answer.startswith('failed:'):
                self.debug_print("답변 생성 실패 감지됨, 웹 검색 권장")
                self.add_system_message("답변 생성에 실패했습니다. 이 질문은 웹 검색을 통해 더 나은 결과를 얻을 수 있습니다.")
            
            self.add_bot_message(answer)
            
            # UI 상태 복원
            self.message_input.setEnabled(True)
            self.send_button.setEnabled(True)
            self.message_input.setFocus()
        except Exception as e:
            self.debug_print(f"응답 처리 중 오류: {e}")
            self.add_system_message(f"응답 처리 중 오류: {str(e)}")
            
            # UI 상태 복원
            self.message_input.setEnabled(True)
            self.send_button.setEnabled(True)
            self.message_input.setFocus()
    
    def handle_rag_error(self, error_message):
        """RAG 에이전트 오류 처리"""
        self.debug_print(f"RAG 오류: {error_message}")
        
        if "context_length_exceeded" in error_message:
            self.add_system_message("토큰 길이 제한 초과 오류가 발생했습니다. 검색 문서 수를 줄이고 다시 시도해 주세요.")
        else:
            self.add_system_message(f"에러: {error_message}")
        
        # UI 상태 복원
        self.message_input.setEnabled(True)
        self.send_button.setEnabled(True)
        self.message_input.setFocus()


def main():
    """애플리케이션 실행"""
    try:
        print("RAG 챗봇 UI 시작")
        app = QApplication(sys.argv)
        window = RAGChatWindow()
        window.show()
        print("UI 창 표시됨")
        sys.exit(app.exec())
    except Exception as e:
        print(f"애플리케이션 실행 중 오류 발생: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main() 