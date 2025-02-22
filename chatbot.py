import streamlit as st
import logging
from langchain_community.vectorstores import Chroma  # ✅ 최신 버전 적용
from langchain_community.embeddings import OpenAIEmbeddings  # ✅ 최신 버전 적용
from langchain_openai import OpenAI  # ✅ OpenAI 관련 최신 패키지 적용

# 📢 로그 설정 (디버깅용)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("debug.log", mode="w", encoding="utf-8"),
    ]
)

logging.info("🚀 Streamlit 챗봇 시작!")

# 🏗️ ChromaDB 초기화
try:
    embeddings = OpenAIEmbeddings()  # ✅ 최신 패키지 적용
    vectorstore = Chroma(embedding_function=embeddings)
    logging.info("✅ ChromaDB 초기화 성공")
except Exception as e:
    logging.error(f"❌ ChromaDB 초기화 실패: {e}")

st.title("LangChain 챗봇")

query = st.text_input("질문을 입력하세요:")

if st.button("검색"):
    logging.debug(f"🔍 사용자 입력: {query}")
    if not query:
        st.warning("질문을 입력하세요!")
    else:
        try:
            result = vectorstore.similarity_search(query)
            logging.info(f"🔍 검색 결과: {result}")
            st.write(result)
        except Exception as e:
            logging.error(f"❌ 검색 실패: {e}")
            st.error("검색 중 오류가 발생했습니다.")
