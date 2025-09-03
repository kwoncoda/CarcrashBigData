"""
🚗 차분해 (車分解) – Hybrid RAG v4 + LangGraph
───────────────────────────────────────────────
• LangChain 0.2  / FAISS + BM25 Hybrid Search (+ Cohere Rerank)
• **LangGraph** : 검색 → 웹 검색 → 결과 후처리 상태머신
• Tool Calling(OpenAI_FUNCTIONS) / Streamlit UI 유지
"""

import os, re, json, streamlit as st
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from rank_bm25 import BM25Okapi
import langgraph as lg
from langgraph.graph import StateGraph

# ───────────────────── 0. ENV & UTIL ─────────────────────
load_dotenv()
BASE_DIR   = Path(__file__).resolve().parent
JSON_PATH  = BASE_DIR / "accident_data_a.json"
FAISS_DIR  = BASE_DIR / "faiss_db"
LABELS     = ["사고유형", "사고 설명", "과실 비율", "사고 링크"]


def korean_ratio(text: str) -> float:
    return len(re.findall(r"[가-힣]", text)) / max(len(text), 1)

def format_response(text: str) -> str:
    lines = []
    for ln in text.split("\n"):
        ln = ln.strip()
        lines.append(f"\n{ln}" if any(lb in ln for lb in LABELS) else ln)
    return "\n".join(lines).strip()

# ───────────────────── 1. VECTOR & BM25 ─────────────────────
@st.cache_resource(show_spinner=False)
def load_vector_and_bm25():
    with open(JSON_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    docs = [Document(page_content=(
        f"사고유형: {c['사고유형']}\n"
        f"자동차 A: {c['자동차 A']}\n"
        f"자동차 B: {c['자동차 B']}\n"
        f"사고 설명: {c['사고 설명']}\n"
        f"과실 비율: {c['과실 비율']}\n"
        f"사고 링크: {c['사고 링크']}")) for c in cases]

    emb = AzureOpenAIEmbeddings(
        deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"))

    if FAISS_DIR.exists():
        vs = FAISS.load_local(str(FAISS_DIR), emb, allow_dangerous_deserialization=True)
    else:
        vs = FAISS.from_documents(docs, emb)
        vs.save_local(str(FAISS_DIR))

    bm25 = BM25Okapi([re.findall(r"\w+", d.page_content.lower()) for d in docs])
    return vs, bm25, docs

vectorstore, bm25_idx, ALL_DOCS = load_vector_and_bm25()

# ───────────────────── 2. HYBRID RETRIEVER ─────────────────────
from langchain_core.retrievers import BaseRetriever

class HybridRetriever(BaseRetriever):
    """FAISS 벡터 검색 + BM25 키워드 검색 결과를 합치는 하이브리드 리트리버."""

    vectorstore: FAISS
    bm25: BM25Okapi
    kv: int = 3  # 벡터 top‑k
    kk: int = 3  # BM25 top‑k

    def _get_relevant_documents(self, query: str):
        vec_docs = self.vectorstore.similarity_search(query, k=self.kv)
        scores   = self.bm25.get_scores(re.findall(r"\w+", query.lower()))
        idx      = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.kk]
        bm_docs  = [ALL_DOCS[i] for i in idx]
        return vec_docs + bm_docs

retriever = HybridRetriever(vectorstore=vectorstore, bm25=bm25_idx)

# ───────────────────── 3. COHERE RERANK (옵션) ─────────────────────
import cohere
co = cohere.Client(os.getenv("COHERE_API_KEY")) if os.getenv("COHERE_API_KEY") else None

def cohere_rerank(query: str, docs: List[Document], top_n: int = 1):
    """(옵션) Cohere Rerank API로 재정렬 후 상위 N개 반환."""
    if not co:
        return docs[:top_n]
    res = co.rerank(query=query, documents=[d.page_content for d in docs], model="rerank-english-v2.0", top_n=top_n)
    return [docs[r.index] for r in res]

# ───────────────────── 4. TOOL 정의 ─────────────────────

@tool
def AccidentSimilaritySearch(query: str) -> dict:
    """DB 유사 판례 1건(4줄 포맷)을 반환."""
    docs = retriever.invoke(query)
    docs = cohere_rerank(query, docs, top_n=1)
    content = "\n\n".join(d.page_content for d in docs) or "(유사 사례 없음)"
    return {"search": content}


@tool
def WebSearch(query: str) -> dict:
    """DuckDuckGo 상위 3개 결과 요약."""
    summary = DuckDuckGoSearchResults(k=3).run(query)
    return {"web": summary}


# ───────────────────── 5. LLM ─────────────────────
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

# ───────────────────── 6. LANGGRAPH ─────────────────────
from langgraph.graph import StateGraph, END
from typing import TypedDict

class GraphState(TypedDict):
    query: str
    search: str
    web: str
    result: str

graph = StateGraph(GraphState)

graph.add_node("search", AccidentSimilaritySearch)
graph.add_node("web", WebSearch)

import re

def clean_web_results(web_text):
    # 링크 제거
    web_text = re.sub(r'https?://\S+', '', web_text)

    # snippet 제거
    web_text = re.sub(r'\bsnippet:[^:]*?:', '', web_text)

    # 번호와 문장 분리
    entries = re.split(r'\(\d+\)', web_text)
    entries = [e.strip() for e in entries if e.strip()]

    # 문장 정리
    sentences = []
    for e in entries:
        e = re.sub(r'\s+', ' ', e).strip()
        if not e.endswith('다.'):
            e += '다.'
        sentences.append(f"- {e}")

    return "\n".join(sentences)


def postprocess(state):
    search = state.get("search", "(검색 결과 없음)")
    web = state.get("web", "(웹 검색 결과 없음)")

    cleaned_web = clean_web_results(web)

    raw = f"(1) 판례 결과\n{search}\n\n(2) 웹 검색 결과\n{cleaned_web}"

    # 번역 비율 검사 후 번역
    if korean_ratio(raw) < 0.7:
        raw = translator.predict(f"한국어로 번역해 형식 유지, 판례를 찾은것은 마지막에 링크를 걸어주고 웹에서 찾은것은 자연스럽게 요약을 해\n\n{raw}")

    return {"result": format_response(raw)}



graph.add_node("format", postprocess)

graph.set_entry_point("search")
graph.add_edge("search", "web")
graph.add_edge("web", "format")
graph.add_edge("format", END)

graph_agent = graph.compile()


# ───────────────────── 7. STREAMLIT UI ─────────────────────
st.set_page_config(page_title="차분해", layout="centered")
st.title("🚗 차분해 (車分解)")
st.write("사고 상황을 입력하면 과실 비율과 관련 설명을 제공합니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user = st.chat_input("사고 상황을 입력해주세요 (예: 후진 중 진행 차량과 충돌 등)")
if user:
    st.chat_message("user").markdown(user)
    st.session_state.messages.append({"role": "user", "content": user})

    if len(user.strip()) < 15:
        warn = "⚠️ 설명이 짧습니다. 장소·속도 등을 포함해 1~2문장으로 입력해주세요."
        st.chat_message("assistant").markdown(warn)
        st.session_state.messages.append({"role": "assistant", "content": warn})
    else:
        with st.spinner("🧠 판례·웹 정보 분석 중..."):
            result_state = graph_agent.invoke({"query": user})
            print("🔍 반환된 상태:", result_state)
            reply = result_state["result"]
        st.chat_message("assistant").markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
