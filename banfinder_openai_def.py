# Streamlit + LangChain (ChatGPT gpt-5) - Image Items Classifier
# ---------------------------------------------------------------
# Quick start
#   pip install streamlit pillow langchain langchain-openai
#   export OPENAI_API_KEY="YOUR_KEY"
#   streamlit run app_items_classifier.py

from __future__ import annotations
import os
import io
import base64
import json
from typing import List
import datetime as dt
import re
import pandas as pd
from rapidfuzz import process, fuzz
import streamlit as st
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import cv2
import numpy as np


def ensure_api_key() -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        st.warning("OPENAI_API_KEY 가 설정되지 않았습니다.")
    return key

def image_to_data_uri(img: Image.Image, format_: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format_)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = f"image/{format_.lower()}"
    return f"data:{mime};base64,{b64}"

def build_system_prompt( depart: str = '', arrive: str = '') -> str:

    df = pd.read_excel('prohibited.xlsx')
    json_str = df.to_json(orient="records", force_ascii=False, indent=2)
    return f"""
    나는 여행을 갈건데, 짐을 체크해볼거야.
    너는 반드시 JSON 형식으로만 답해야 해. 
    리스트 4개 요소를 차례대로 채워줘.

    구조 예시:
    [
      "첫번째 섹션 내용",
      "두번째 섹션 내용",
      "세번째 섹션 내용"
    ]

    규칙:
    - 한글로 답해줘.
    - 빠진 물품도 추천해줘.
    - 이동수단은 비행기야.
    - 물품분류는 최대한 상세히 해줘.
    - 답변할 때 보기 편하게 개행을 해줘.
    - 내용이 없어도 4가지가 나오게 해줘
    - 반입금지물품은 기내, 수하물로 구분되어 있는데 다음 JSON 데이터야.: {json_str}
    - 해당되는 물품 중 '비고'가 있으면 같이 알려주고, 해당되지 않으면 생략
    - 식별 정확도가 현저히 떨어지는 물품은 추측하지말고 패스해.

    네가 채워야 할 3개 리스트 순서는 다음과 같아:
    1. 물품분류(상세히 작성)
    2. 반입금지물품(물품분류 리스트에 따라서 작성)
    3. 빠진물품추천
    """



def analyze_image_with_yolo(image, model):
    """
    Streamlit에서 업로드된 이미지 파일을 받아 YOLO 모델로 분석하고 결과 이미지 반환하는 함수.

    Args:
        uploaded_file: Streamlit file_uploader에서 받은 이미지 파일 객체
        model: YOLO 모델 인스턴스 (예: YOLO('yolov8n.pt'))

    Returns:
        result_img: 분석 결과가 시각화된 OpenCV 이미지 (numpy array)
    """
    
    # 2. PIL 이미지를 OpenCV 형식으로 변환 (RGB -> BGR)
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # 3. YOLO 모델로 예측 수행
    results = model(img_bgr)
    
    # 4. 결과 이미지(주석 포함) 얻기 (OpenCV BGR 이미지)
    result_img = results[0].plot()
    
    return result_img


def classify_items(img: Image.Image, depart: str = '', arrive: str = '') -> str:
    """Return a natural, conversational Korean description (not JSON)."""
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.0)
    data_uri = image_to_data_uri(img)

    system = SystemMessage(content=build_system_prompt(depart, arrive))
    user_content = [
        {
            "type": "text",
            "text": 
                "이미지 안에 어떤 물품(객체)들이 있는지 설명해줘. "
        },
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]


    resp = llm.invoke([system, HumanMessage(content=user_content)])

    usage = resp.response_metadata.get("token_usage", {})
    print("프롬프트 토큰:", usage.get("prompt_tokens"))
    print("출력 토큰:", usage.get("completion_tokens"))
    print("총 사용 토큰:", usage.get("total_tokens"))

    
    return resp.content  # 자연어 한국어 설명

# def normalize_text(s: str) -> str:
#     s = (s or "").strip().lower()
#     s = re.sub(r"[^\w가-힣\s]", " ", s)
#     s = re.sub(r"\s+", " ", s).strip()
#     return s

# def split_blocks(text: str) -> list:
#     BLOCK_RE = re.compile(r"---section---")
#     # ---SECTION--- 을 기준으로 자르고, 공백 블록은 제거
#     parts = BLOCK_RE.split(text)
#     return [p.strip() for p in parts if p.strip()]