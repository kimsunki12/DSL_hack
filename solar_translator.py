import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
from dotenv import load_dotenv

# .env 파일 로드 (환경 변수가 설정되지 않았을 경우 대비)
load_dotenv()

# FastAPI 앱 초기화
app = FastAPI()

# UPSTAGE_API_KEY 가져오기 (Solar LLM API 키)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("🚨 환경 변수 'UPSTAGE_API_KEY'가 설정되지 않았습니다!")

# Solar LLM 모델 설정 (API 키 적용)
solar_llm = ChatOpenAI(model="gpt-4o", temperature=0.7, openai_api_key=api_key)

# 프롬프트 템플릿 설정
prompt = ChatPromptTemplate.from_template("Translate the following text to {target_language}: {text}")

# LangChain 체인 설정
chain = prompt | solar_llm

# LangChain 체인을 FastAPI 엔드포인트로 추가
add_routes(app, chain, path="/translate")

# API 요청 모델 정의
class TranslationRequest(BaseModel):
    text: str
    target_language: str

# 번역 API 엔드포인트
@app.post("/translate")
async def translate_text(request: TranslationRequest):
    """사용자가 입력한 텍스트를 원하는 언어로 번역"""
    result = await chain.ainvoke({"text": request.text, "target_language": request.target_language})
    return {"translated_text": str(result)}

# FastAPI 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
