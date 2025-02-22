import pysqlite3
pysqlite3.install_as_sqlite3()

import os
import streamlit as st
import base64
import uuid
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ✅ 환경 변수 로드
load_dotenv()

# ✅ 세션 상태 초기화 (기존 값 유지)
if "id" not in st.session_state:
    st.session_state.id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

session_id = st.session_state.id

@st.cache_resource
def process_pdf(file_path: str):
    """📌 PDF 파일을 처리하여 retriever 및 rag_chain을 생성"""
    if not os.path.exists(file_path):
        st.error("❌ PDF 파일이 존재하지 않습니다.")
        return None, None

    st.info("📖 PDF 파일을 분석 중... 잠시만 기다려주세요!")
    
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    # ✅ 벡터스토어 및 retriever 생성
    vectorstore = Chroma.from_documents(pages, UpstageEmbeddings(model="solar-embedding-1-large"))
    retriever = vectorstore.as_retriever(k=3)

    chat = ChatUpstage(upstage_api_key=os.getenv("UPSTAGE_API_KEY"), model="solar-pro")

    # ✅ Contextualized 질문 생성
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "이전 대화 내용을 참고하여 사용자의 질문을 독립적으로 이해할 수 있도록 다시 구성해주세요."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(chat, retriever, contextualize_q_prompt)

    # ✅ 질문-답변 체인 생성
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 문서를 참고하여 질문에 대한 답을 생성하세요.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return retriever, rag_chain

def save_uploaded_file(uploaded_file):
    """📌 업로드된 파일을 저장하고, 파일 경로를 반환"""
    file_key = f"{session_id}-{uploaded_file.name}"
    save_path = os.path.join("uploaded_files", file_key)

    if not os.path.exists("uploaded_files"):
        os.makedirs("uploaded_files")

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    return save_path

def display_pdf(file_path):
    """📌 PDF 파일을 웹 페이지에서 미리보기"""
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    st.markdown("### PDF 미리보기")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" 
                        width="100%" height="600px" type="application/pdf"></iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)

# ✅ 사이드바 설정
with st.sidebar:
    st.header("📄 PDF 파일 업로드")
    uploaded_file = st.file_uploader("Choose a .pdf file", type="pdf")

    if uploaded_file:
        file_path = save_uploaded_file(uploaded_file)

        # ✅ 같은 파일이면 재처리 방지
        if st.session_state.get("current_pdf") == file_path:
            st.info("📌 이미 업로드된 PDF입니다.")
        else:
            st.session_state["current_pdf"] = file_path
            st.session_state.file_uploaded = True  # ✅ 파일 업로드 상태 유지
            display_pdf(file_path)

            retriever, rag_chain = process_pdf(file_path)

            if retriever and rag_chain:
                st.session_state.retriever = retriever
                st.session_state.rag_chain = rag_chain  # ✅ 세션 유지
                st.success("✅ RAG 모델이 정상적으로 로드되었습니다!")
            else:
                st.error("❌ RAG 모델 로드 실패! 로그를 확인하세요.")

        st.success("✅ 파일이 성공적으로 업로드되었습니다!")

# ✅ 웹사이트 제목
st.title("Solar LLM Chatbot")

# ✅ 파일이 업로드된 후 세션이 유지되는지 확인
if st.session_state["file_uploaded"]:
    st.success(f"📂 현재 PDF: {os.path.basename(st.session_state['current_pdf'])}")
else:
    st.warning("⚠️ PDF 파일을 업로드해주세요.")

# ✅ 입력창이 사라지는 문제 해결
prompt = st.chat_input("💬 PDF에 대해 질문해보세요!")

if prompt and st.session_state.get("rag_chain"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        result = st.session_state["rag_chain"].invoke({"input": prompt, "chat_history": st.session_state.messages})

        with st.expander("📖 참고 문서"):
            st.write(result["context"])

        for chunk in result["answer"].split(" "):
            full_response += chunk + " "
            time.sleep(0.2)
            message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

# ✅ **입력창이 무조건 보이도록 보완**
if not st.session_state.get("rag_chain"):
    st.info("📌 PDF가 정상적으로 처리되면 질문을 입력할 수 있습니다.")
