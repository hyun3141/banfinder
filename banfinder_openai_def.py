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
import time
from google import genai
from google.genai import types


JSON_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "sections": types.Schema(
            type=types.Type.ARRAY,
            description="이미지 분석 결과를 담는 2개의 섹션 문자열 배열입니다.",
            items=types.Schema(
                type=types.Type.STRING,
                description="각 섹션의 내용 (1: 반입금지물품, 2: 물품분류)"
            )
        )
    },
    required=["sections"]
)


def ensure_api_key_openai() -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        st.warning("OPENAI_API_KEY 가 설정되지 않았습니다.")
    return key


def ensure_api_key_gemini() -> str | None:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        st.warning("GEMINI_API_KEY 가 설정되지 않았습니다.")
    return key

def image_to_data_uri(img: Image.Image, format_: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format_)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    # padding이 없거나 부족하면 '=' 추가
    padding = 4 - (len(b64) % 4)
    if padding and padding != 4:
        b64 += '=' * padding

    mime = f"image/{format_.lower()}"
    return f"data:{mime};base64,{b64}"


def build_system_prompt( depart: str = '', arrive: str = '') -> str:

    df = pd.read_excel('prohibited.xlsx')
    json_str = df.to_json(orient="records", force_ascii=False, indent=2)
    return f"""
    나는 여행을 갈건데, 짐을 체크해볼거야.
    리스트 2개 요소를 차례대로 채워줘.

    구조 예시:
    [
      "첫번째 섹션 내용",
      "두번째 섹션 내용"
    ]

    규칙:
    - 한글로 답해줘. json형식으로 답해줘.
    - 이동수단은 비행기야.
    - 답변할 때 보기 편하게 개행을 해줘.
    - 반입금지물품은 기내, 수하물로 구분되어 있는데 다음 JSON 데이터야.: {json_str}
    - 해당되는 물품 중 '비고'가 있으면 같이 알려주고, 해당되지 않으면 생략
    - 식별 정확도가 현저히 떨어지는 물품은 추측하지말고 패스해.
    - 불필요한 말은 하지말아주고 최대한 간략히 해줘.
    - 반입금지물품은 금지사항이 있을경우에만 기내/위탁 가능여부 O/X만 알려주고, 비고가 있을 경우 작성
    
    네가 채워야 할 2개 리스트 순서는 다음과 같아:
    1. 반입금지물품
    2. 물품분류
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


def classify_items_openai(img: Image.Image, depart: str = '', arrive: str = '') -> str:
    """Return a natural, conversational Korean description (not JSON)."""
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.0, reasoning_effort="minimal", verbosity="high")
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


    start_time = time.time()
    resp = llm.invoke([system, HumanMessage(content=user_content)])
    end_time = time.time()

    usage = resp.response_metadata.get("token_usage", {})
    print("프롬프트 토큰:", usage.get("prompt_tokens"))
    print("출력 토큰:", usage.get("completion_tokens"))
    print("총 사용 토큰:", usage.get("total_tokens"))
    print("응답에 걸린 시간:", end_time - start_time, "초")

    
    return resp.content  # 자연어 한국어 설명


def classify_items_gemini(img: Image.Image, depart: str = '', arrive: str = '') -> str:
    """Return a natural, conversational Korean description (not JSON) using Google Gemini."""

    client = genai.Client()

    # 이미지 데이터를 base64 등 바이너리로 준비
    img_data = image_to_data_uri(img)
    base64_str = img_data.split(",")[1]
    image_bytes = base64.b64decode(base64_str)
    
    
    # Google Gemini는 멀티모달 입력을 리스트 형태로 받음 (이미지 + 텍스트)
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                types.Part.from_text(text=build_system_prompt(depart, arrive)),
            ]
        )
    ]



    config = types.GenerateContentConfig(
        temperature=0,    # 창의성 수준 (0~2)
        top_p=0,          # 확률 분포 컷오프 (기본 1.0)
        response_mime_type="application/json",
        response_schema=JSON_RESPONSE_SCHEMA, # 👈 스키마 객체 추가
        max_output_tokens=10000,
        # 추가 파라미터 가능
    )
    start_time = time.time()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=config,
    )
    end_time = time.time()
    

    try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=config,
            )
            end_time = time.time()
        
            print("응답에 걸린 시간:", end_time - start_time, "초")

            usage = response.usage_metadata
            if usage:
                print("프롬프트 토큰:", usage.prompt_token_count)
                print("출력 토큰:", usage.candidates_token_count)
                print("총 사용 토큰:", usage.total_token_count)

            # response.text에 자연어 설명이 담겨 있음
            return response.text
    
    except Exception as e:
        print(f"🚨 Gemini API 호출 중 오류 발생: {e}")
        # 오류 발생 시 빈 JSON 리스트 문자열을 반환하여 UI 코드의 json.loads 오류 방지
        return '["API 호출 오류", "API 호출 오류"]' # 유효한 JSON 문자열 반환



