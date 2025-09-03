import requests
from bs4 import BeautifulSoup
import json
import time
import urllib3

# 데이터 저장 리스트
accident_data = []

for page_number in range(1, 62):
    sub_page_number = 1
    
    while True:
        # 사고유형 코드 생성
        chart_code = f"차{page_number}-{sub_page_number}"
        encoded_chart_code = requests.utils.quote(chart_code, encoding="utf-8")
        url = f"https://accident.knia.or.kr/myaccident-content?chartNo={encoded_chart_code}&chartType=1"

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(url, verify=False)
        print(url)

        if "요청하신 페이지를 찾을 수 없습니다" in response.text:
            break

        soup = BeautifulSoup(response.text, "html.parser")

        car_A = soup.select_one(".cont_l .con")
        car_B = soup.select_one(".cont_r .con")
        accident_description = soup.select_one("#smrizeexplna")
        fault_A = soup.select_one("td .red")
        fault_B = soup.select_one("td .orange")

        if car_A and car_B and accident_description and fault_A and fault_B:
            accident_data.append({
                "사고유형": chart_code,
                "자동차 A": car_A.text.strip(),
                "자동차 B": car_B.text.strip(),
                "사고 설명": accident_description.text.strip().replace("\r\n", " ").replace("\n", " ").replace("⊙ ", ""),
                "과실 비율": f"{fault_A.text.strip()} : {fault_B.text.strip()}",
                "사고 링크": url  # 이 부분이 핵심입니다
            })

        sub_page_number += 1
        time.sleep(1)

# JSON 파일 저장
with open("accident_data_a.json", "w", encoding="utf-8") as f:
    json.dump(accident_data, f, ensure_ascii=False, indent=4)

print("✅ 크롤링 완료! 사고 링크가 포함된 accident_data.json 생성됨")
