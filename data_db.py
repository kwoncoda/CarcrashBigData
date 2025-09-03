# streamlit_app.py (LangChain **Agent** – 한국어 강화형 v3)
"""
🚗 차분해 (車分解)
- 사고 사례 유사도 검색 + 보완 검색
- DB 검색 + Web 검색 결과 모두 사용
- DB 결과는 고정 형식으로, Web은 자연어로 보완 설명
- 최종 출력은 반드시 한국어
"""

import os, json, re, streamlit as st
from dotenv import load_dotenv
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.tools import DuckDuckGoSearchResults

# ===================== 환경 & 유틸 =====================
load_dotenv()

def get_api_key():
    if not (os.getenv("OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT")):
        st.error(".env 파일에 OPENAI_API_KEY 또는 AZURE_OPENAI_ENDPOINT가 누락되었습니다.")
        st.stop()

def korean_ratio(text: str) -> float:
    korean = re.findall(r"[가-힣]", text)
    return len(korean) / max(len(text), 1)

LABELS = ["사고유형", "사고 설명", "과실 비율", "참고 링크"]

def format_response(txt: str) -> str:
    out = []
    for ln in txt.split("\n"):
        ln = ln.strip()
        out.append(f"\n{ln}" if any(lb in ln for lb in LABELS) else ln)
    return "\n".join(out).strip()

# ===================== 벡터스토어 =====================
@st.cache_resource
def load_vectorstore():
    with open("accident_data_a.json", "r", encoding="utf-8") as f:
        cases = json.load(f)
    docs: List[Document] = [
        Document(
            page_content=(
                f"사고유형: {c['사고유형']}\n"
                f"자동차 A: {c['자동차 A']}\n"
                f"자동차 B: {c['자동차 B']}\n"
                f"사고 설명: {c['사고 설명']}\n"
                f"과실 비율: {c['과실 비율']}\n"
                f"사고 링크: {c['사고 링크']}"
            )
        )
        for c in cases
    ]
    emb = AzureOpenAIEmbeddings(
        deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    return FAISS.from_documents(docs, emb)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# ===================== LLM =====================
llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.3,
)

translator = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.2,
)

# ===================== 도구 =====================
def accident_similarity_search(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs) or "(유사 사례 없음)"

vector_tool = Tool(
    name="AccidentSimilaritySearch",
    func=accident_similarity_search,
    description="사고 설명에 대해 유사 사례를 검색한다."
)

web_tool = Tool(
    name="WebSearch",
    func=DuckDuckGoSearchResults().run,
    description="사고 설명을 바탕으로 보완 정보를 검색한다."
)

# ===================== 에이전트 시스템 프롬프트 =====================
SYSTEM_PROMPT = """
너는 교통사고 과실비율 전문가 챗봇이다.
1. 사고 설명을 받으면 AccidentSimilaritySearch 도구로 유사한 판례 1건을 찾아 사고유형, 설명, 과실 비율, 링크 4가지 항목으로 보여준다.
2. 이후 동일 설명으로 WebSearch 도구를 호출하여 사고 내용에 대한 일반적 해설을 추가한다.
3. 출력은 반드시 다음 순서를 따른다:
  (1) 판례 결과
  (2) 웹 검색 결과 요약 (자연스러운 한글 문단)
4. 전체 출력이 한국어가 아니거나 한국어 비율이 낮을 경우 자동 번역한다.
5. 반드시 한국어로만 답변하며 인사말, 영어, 코드블럭, 마크다운은 금지한다.

"""

agent = initialize_agent(
    tools=[vector_tool, web_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_kwargs={"system_message": SYSTEM_PROMPT},
    verbose=False,
)

# ===================== Streamlit UI =====================
st.set_page_config(page_title="차분해 - 사고 과실비율 챗봇", layout="centered")
st.title("🚗 차분해 (車分解)")
st.markdown("사고 상황을 입력하면 과실 비율과 관련 설명을 함께 제공합니다.")
get_api_key()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user = st.chat_input("사고 상황을 입력해주세요. (예: 후진 중 진행 차량과 충돌 등)")
if user:
    st.chat_message("user").markdown(user)
    st.session_state.messages.append({"role": "user", "content": user})

    if len(user.strip()) < 15:
        warn = "⚠️ 설명이 너무 짧습니다. 장소, 방향, 속도 등을 포함한 1~2문장으로 자세히 입력해주세요."
        st.chat_message("assistant").markdown(warn)
        st.session_state.messages.append({"role": "assistant", "content": warn})
    else:
        with st.spinner("🧠 유사 판례 및 웹 정보를 분석 중입니다..."):
            raw = agent.run(user)
            if korean_ratio(raw) < 0.7:
                translate_prompt = "다음 내용을 한국어로 번역하되, 형식을 유지하세요: 1. 사고 설명을 받으면 AccidentSimilaritySearch 도구로 유사한 판례 1건을 찾아 사고유형, 설명, 과실 비율, 링크 4가지 항목으로 보여준다. 2. 이후 동일 설명으로 WebSearch 도구를 호출하여 사고 내용에 대한 일반적 해설을 추가한다. 3. 출력은 반드시 다음 순서를 따른다:  (1) 판례 결과  (2) 웹 검색 결과 요약 (자연스러운 한글 문단) 4. 전체 출력이 한국어가 아니거나 한국어 비율이 낮을 경우 자동 번역한다. 5. 반드시 한국어로만 답변하며 인사말, 영어, 코드블럭, 마크다운은 금지한다.\n\n" + raw
                raw = translator.predict(translate_prompt)
            reply = format_response(raw)
        st.chat_message("assistant").markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
