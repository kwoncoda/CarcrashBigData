import os
import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(BASE_DIR, "data/conversion", "과실비율_v1.md")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/conversion", "토큰_분석_출력.md")

# Markdown 내용 불러오기
with open(MD_PATH, "r", encoding="utf-8") as f:
    markdown_text = f.read()
    
    
# Document 객체로 감싸기
doc = Document(page_content=markdown_text)
   
# TikToken 인코더를 사용하여 재귀적 텍스트 분할기 초기화
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    #encoding_name="cl100k_base",
    model_name="gpt-4o-mini",
    chunk_size=300,
    chunk_overlap=20,
    separators=["\n\n", "\n", ".", " ", ""],
)
chunks = text_splitter.split_documents([doc])


tokenizer = tiktoken.encoding_for_model("gpt-4o")


# 전체 청크 내용 저장
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        tokens = tokenizer.encode(chunk.page_content)
        token_count = len(tokens)

        f.write(f"[Chunk {i+1}] ({token_count} tokens)\n")
        f.write(chunk.page_content.strip() + "\n")

print(f"✅ 토큰 분석 결과가 저장되었습니다: {OUTPUT_PATH}")