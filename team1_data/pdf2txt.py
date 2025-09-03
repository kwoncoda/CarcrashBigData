import os
import pdfplumber
import re


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  


def clean_cid(text):
    return re.sub(r"\(cid:\d+\)", "", text)


def pdfconversion(pdf_path, output_path):
    
    PDF_PATH = os.path.join(BASE_DIR, "data/original", pdf_path)
    OUTPUT_PATH = os.path.join(BASE_DIR, "data/conversion", output_path)

    with pdfplumber.open(PDF_PATH) as pdf, open(OUTPUT_PATH, "w", encoding="utf-8") as out_file:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                cleaned = clean_cid(text)
                out_file.write(cleaned)
            else:
                out_file.write("[텍스트 없음]")
            
            
    print(f"텍스트가 '{OUTPUT_PATH}' 파일로 저장되었습니다.")


pdfconversion("231107_과실비율인정기준_온라인용 (1)-삭제된 페이지.pdf","과실비율_v1.md")
pdfconversion("231107_과실비율인정기준_온라인용.pdf","과실비율_v2.md")