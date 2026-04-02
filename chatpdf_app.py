import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# 환경 설정 및 API 키 로드
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    st.error('OPENAI_API_KEY를 .env 파일에 설정해주세요.')
    st.stop()

# ---------------------------------------------------------
# 🎨 1. 화면 기본 설정 (layout="wide"로 넓게 쓰기 가능)
st.set_page_config(page_title="PDF AI 챗봇", page_icon="📚", layout="centered")

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = None

# ---------------------------------------------------------
# 🎨 2. 메인 타이틀 및 환영 문구
st.title("📚 PDF AI 챗봇")
st.markdown("궁금한 PDF 문서를 올리고 무엇이든 물어보세요! AI가 문서를 읽고 답변해 드립니다.")
st.divider() # 가로 구분선

# ---------------------------------------------------------
# 🎨 3. 사이드바 예쁘게 꾸미기
with st.sidebar:
    st.header("📂 문서 업로드")
    st.caption("분석하고 싶은 PDF 파일을 여기에 올려주세요.")
    uploaded = st.file_uploader("", type="pdf")
    
    st.divider()
    
    # 아코디언 메뉴 (사용 팁)
    with st.expander("💡 사용 팁"):
        st.markdown(
            """
            - **질문은 구체적으로!** (예: 주인공 이름이 뭐야?)
            - 문서에 없는 내용은 AI가 대답하지 않습니다.
            - 새 문서를 올리면 기존 대화가 초기화됩니다.
            """
        )
    
    # 벡터 DB 생성 로직 (기존과 동일)
    if uploaded is not None and st.session_state.chain is None:
        with st.spinner("로봇이 PDF를 열심히 읽고 있습니다... 🤖"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded.getvalue())
                tmp_file_path = tmp_file.name
            
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(pages)
            
            embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
            vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
            
            retriever = vectorstore.as_retriever(search_kwargs={'k': 8})
            llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
            
            prompt = ChatPromptTemplate.from_messages([
                ('system',
                 '당신은 친절하고 스마트한 PDF 문서 분석 전문가입니다.\n'
                 '아래 문서 내용을 바탕으로만 질문에 답하세요.\n'
                 '문서에 없는 내용은 "문서에서 확인되지 않습니다"라고 정중하게 답하세요.\n\n'
                 '[문서 내용]\n{context}'),
                MessagesPlaceholder(variable_name='chat_history'),
                ('human', '{question}'),
            ])
            
            def format_docs(docs):
                return '\n\n'.join(
                    f'[페이지 {doc.metadata.get("page", i)+1}] {doc.page_content}'
                    for i, doc in enumerate(docs)
                )
                
            chain = (
                {
                    'context': (lambda x: x['question']) | retriever | format_docs,
                    'question': lambda x: x['question'],
                    'chat_history': lambda x: x.get('chat_history', []),
                }
                | prompt | llm | StrOutputParser()
            )
            
            st.session_state.chain = chain
            st.success("✅ 문서 분석 완료!")

# ---------------------------------------------------------
# 질문 처리 함수
def process_question(user_question):
    # 🎨 4. 사용자 아바타 적용
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(user_question)
    st.session_state.chat_history.append(HumanMessage(content=user_question))
    
    # 🎨 4. AI 아바타 적용
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("답변을 생성 중입니다..."):
            answer = st.session_state.chain.invoke({
                'question': user_question,
                'chat_history': st.session_state.chat_history
            })
            st.write(answer)
    st.session_state.chat_history.append(AIMessage(content=answer))

# ---------------------------------------------------------
# 🎨 5. 초기 환영 메시지 (대화가 없을 때)
if len(st.session_state.chat_history) == 0:
    with st.chat_message("assistant", avatar="🤖"):
        st.write("안녕하세요! 저는 PDF 문서 분석 봇입니다. 왼쪽 사이드바에서 PDF를 업로드한 뒤 질문을 남겨주세요!")

# 화면에 기존 대화 기록 출력
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(msg.content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(msg.content)

# 하단 입력창 및 조건부 실행
if st.session_state.chain is not None:
    if user_input := st.chat_input("이 문서에 대해 무엇이든 물어보세요!"):
        process_question(user_input)
else:
    st.info("👈 왼쪽 사이드바에서 PDF 파일을 먼저 업로드해 주세요.")