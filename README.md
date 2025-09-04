# 차분해 (車分解)

## 프로젝트 개요
자동차 사고 상황을 입력하면 유사한 판례와 웹 검색 결과를 조합하여 과실 비율을 안내하는 하이브리드 RAG 기반 챗봇입니다.

## 소개
이 프로젝트는 FAISS 벡터 검색과 BM25 키워드 검색을 결합하고, 필요 시 Cohere 재정렬과 DuckDuckGo 웹 검색을 통해 사고 관련 정보를 제공합니다. LangGraph 상태 머신과 Streamlit UI를 통해 한국어 대화형 인터페이스를 구현했습니다.

## 상세 기능
- **유사 판례 검색**: 사고 데이터를 벡터화하여 FAISS와 BM25를 함께 사용해 관련 판례를 탐색합니다.
- **웹 보완 검색**: DuckDuckGo에서 상위 결과를 요약하여 판례 정보를 보완합니다.
- **결과 후처리 및 번역**: 검색 결과를 정리하고 한국어 비율이 낮으면 자동 번역합니다.
- **대화형 UI**: Streamlit 기반 채팅 인터페이스로 사용자의 입력과 답변을 관리합니다.

## 상세 기술
- LangChain 0.2, LangGraph 기반 상태 머신
- FAISS + BM25 하이브리드 검색 및 Cohere Rerank(옵션)
- Azure OpenAI Embeddings & Chat 모델 사용
- DuckDuckGo Search API, Streamlit UI

## 주요 구성요소
- `app.py`: LangGraph를 사용한 메인 Streamlit 애플리케이션
- `data_db.py`: LangChain 에이전트를 사용한 대안 구현
- `accident_data_a.json`: 사고 사례 데이터셋
- `faiss_db/`: 사전 구축된 FAISS 인덱스
- `pyproject.toml`: 필요한 패키지 및 Python 버전 정의

## 실행 전 준비
1. Python 3.10 이상 설치
2. `uv`로 가상환경 생성 및 의존성 설치
   ```bash
   uv sync

   ```
3. `.env` 파일에 다음 환경 변수를 설정
   - `OPENAI_API_KEY`
   - `AZURE_OPENAI_ENDPOINT`
   - `AZURE_OPENAI_CHAT_DEPLOYMENT`
   - `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
   - `AZURE_OPENAI_API_VERSION`
   - (선택) `COHERE_API_KEY`

## 실행 과정
```bash
uv run streamlit run app.py
```
브라우저에서 나타나는 인터페이스에 사고 상황을 입력하면 유사 판례와 웹 검색 결과가 반환됩니다.
