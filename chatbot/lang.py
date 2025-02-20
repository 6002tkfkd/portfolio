from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from dotenv import load_dotenv
import os
import time
import traceback

load_dotenv()

class ChatbotManager:
    def __init__(self):
        self.llm = None
        self.retriever = None
        self.prompt = None
        self.initialized = False
        self.vectorstore = None
        self.last_request_time = 0
        self.min_request_interval = 3  # 최소 3초 간격

    def _wait_for_rate_limit(self):
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def initialize(self):
        if self.initialized:
            return True

        try:
            # API 키 확인
            api_key = os.getenv('MISTRAL_API_KEY')
            if not api_key:
                raise ValueError("MISTRAL_API_KEY not found in environment variables")

            print(f"API Key exists: {bool(api_key)}")

            # Mistral API 설정
            self.llm = ChatMistralAI(
                api_key=api_key,
                model_name="mistral-medium",
                temperature=0.7,
                max_tokens=1024,
                retry_on_rate_limit=True,
                rate_limit=1
            )
            print("LLM initialized")

            # PDF 파일 경로 확인
            pdf_path = os.path.join(os.getcwd(), "chatbot", "hansarang.pdf")
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found at {pdf_path}")

            # PDF 로드 및 분할
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=50
            )
            split_documents = text_splitter.split_documents(docs)

            # 벡터 저장소 설정
            faiss_index_path = "chatbot/vector_store"
            
            if os.path.exists(faiss_index_path):
                print("Loading existing vector store...")
                embeddings = MistralAIEmbeddings(api_key=api_key)
                self.vectorstore = FAISS.load_local(
                    faiss_index_path, 
                    embeddings,
                    allow_dangerous_deserialization=True  # 안전한 환경에서만 사용
                )
            else:
                print("Creating new vector store...")
                embeddings = MistralAIEmbeddings(api_key=api_key)
                processed_documents = []
                
                for i, doc in enumerate(split_documents):
                    if i > 0 and i % 3 == 0:
                        time.sleep(2)
                    processed_documents.append(doc)
                
                self.vectorstore = FAISS.from_documents(documents=processed_documents, embedding=embeddings)
                os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
                self.vectorstore.save_local(faiss_index_path)

            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 2}
            )
            print("Retriever initialized")

            # 프롬프트 템플릿 설정
            self.prompt = PromptTemplate.from_template(
                """당신은 한국어로 대답하는 AI 챗봇입니다.
                다음 내용을 바탕으로 짧고 간결하고 자연스러운 한국어로만 답변해주세요.

                참고 내용: {context}

                질문: {question}

                답변:"""
            )
            print("Prompt template initialized")

            self.initialized = True
            return True

        except Exception as e:
            print(f"Error initializing chatbot: {str(e)}")
            print(traceback.format_exc())
            return False

    def get_response(self, question):
        try:
            # 문서 검색
            self._wait_for_rate_limit()
            docs = self.retriever.get_relevant_documents(question)
            context = "\n".join([doc.page_content for doc in docs])
            
            # LLM 요청
            self._wait_for_rate_limit()
            response = self.prompt.format(context=context, question=question)
            answer = self.llm.invoke(response)
            
            if hasattr(answer, 'content'):
                return answer.content
            return str(answer)
            
        except Exception as e:
            print(f"Error getting response: {str(e)}")
            if "429" in str(e):
                return "죄송합니다. 현재 요청이 많아 잠시 후에 다시 시도해주세요."
            return "죄송합니다. 일시적인 오류가 발생했습니다."

# 전역 인스턴스 생성
chatbot = ChatbotManager()
