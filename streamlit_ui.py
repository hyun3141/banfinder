# Streamlit + LangChain (ChatGPT gpt-5) - Image Items Classifier
# ---------------------------------------------------------------
# Quick start
#   pip install streamlit pillow langchain langchain-openai
#   export OPENAI_API_KEY="YOUR_KEY"
#   streamlit run app_items_classifier.py

from __future__ import annotations
import os
import io
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
from banfinder_openai_def import *
from weather_module import *
from rag_def import *
from ultralytics import YOLO
import cv2
import numpy as np

def main() -> None:


    with st.sidebar:
        st.title("✈️러기지 체크 앱")
        page = st.radio("", ["짐찍기", "챗봇"], index=0, key="page")


    st.set_page_config(page_title="공항앱", page_icon="✈️", layout="centered")

    ensure_api_key()
    # 경유지 리스트 초기화

    desc = []
    result_img_rgb = None
    result_list = []
    
    if page == "짐찍기":

        st.title("✈️사진을 올려주세요")
        if "stops" not in st.session_state:
            st.session_state.stops = []  # ['서울역', '대전역', ...]

        # 날짜 정하기 컨테이너
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("출발")
                depart = st.text_input("출발지", placeholder="예: 서울", value="서울")

            with col2:
                st.subheader("도착")
                arrive = st.text_input("도착지", placeholder="예: 도쿄", value="도쿄")

            with col3:
                st.subheader("✈️날짜")
                arrive_date = st.date_input("날짜", key="date", value=dt.date.today())


        # 이미지업로드 컨테이너
        with st.container():
                
            photo = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg", "webp"], key="upload_image")
            img = None

            if photo is not None:
                img = Image.open(photo).convert("RGB")
                st.image(img, caption=photo.name, use_container_width=True)
                try:
                    result = classify_items(img, depart, arrive)  # 당신이 이미 가진 함수 사용
                    st.subheader("결과")
                    # 대화형(자연어) 응답이라면 그대로 표시
                    
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["반입금지물품(OPENAI)  ", "반입금지물품(YOLO)", "물품분류  ", "빠진물품추천  ", "날씨  "])


                    
                    result_list = json.loads(result)
                    #result_list = [1,1,1,1]
                    with tab1:
                        st.write(result_list[1])

                    with tab2:
                        model = YOLO('yolo_fine_tuned_best.pt')
                        result_yolo = model(img)

                        class_names = result_yolo[0].names  # 클래스 이름 리스트

                        desc = []
                        for box in result_yolo[0].boxes:
                            class_idx = int(box.cls)
                            label = class_names[class_idx]

                            label_str = label.capitalize()

                            if label_str=='Fan':
                                label_desc = '[휴대용 선풍기] - 휴대용선풍기의 경우 기내/수화물 반입이 가능합니다.\n' \
                                +'다만 배터리의 용량을 참고하여야합니다.\n'\
                                +'100wh이하 시 : 인당 5개까지 객실 반입가능(수하물 반입불가), 6개 이상부터는 항공사 승인 필요\n'\
                                +'160wh이하 시 : 인당 2개까지 객실 반입가능(수하물 반입불가), 3개 이상부터는 항공사 승인 필요\n'\
                                +'160wh초과 시 : 객실/수하물 반입불가\n\n'

                                if label_desc not in desc:
                                    desc.append(label_desc)
                            elif label_str=='Spray':
                                label_desc = '[스프레이] - 인화성일 시 객실/수하물 반입불가하며, 그 외 100ml 용량까지 반입가능\n\n'
                                if label_desc not in desc:
                                    desc.append(label_desc)
                            elif label_str=='Lighter':
                                label_desc = '[라이터] - 일회용 또는 지포라이터의 경우 1인 1개까지 객실반입만 가능\n'\
                                +'그외 총기모양 라이터, 대형라이터 객실/수하물 반입불가\n\n'

                                if label_desc not in desc:
                                    desc.append(label_desc)
                            elif label_str=='Iron':
                                label_desc = '[다리미] - 다리미는 일반적으로 위탁 수하물로 부치는 것은 가능하나,\n'\
                                +'기내 탑승 수하물로는 제한될 수 있음.\n'\
                                +'다리미는 무기로 간주될 수 있고, 특히 배터리 내장형 무선 다리미는 추가 규제를 받을 수 있으므로\n'\
                                +'기내 휴대는 항공사 및 여행 목적지에 따라 다르니 사전에 항공사 규정을 확인하는 것이 안전함\n\n'

                                if label_desc not in desc:
                                    desc.append(label_desc)
                            elif label_str=='Buntan_gas':
                                label_desc = '[부탄가스] - 부탄/프로판/이소가스의 경우 기내 및 수하물 반입 금지\n\n'
                                
                                if label_desc not in desc:
                                    desc.append(label_desc)

                        result_img = analyze_image_with_yolo(img, model)
                        
                        # OpenCV 이미지(BGR)를 RGB로 변환해서 스트림릿에 출력
                        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        st.image(result_img_rgb, caption="분석 결과", use_container_width=True)

                        desc_str = ''.join(str(i) for i in desc)

                        st.write(desc_str)


                    with tab3:
                        st.write(result_list[0])

                    with tab4:
                        st.write(result_list[2])

                    with tab5:
                        st.subheader("도착지 날씨")
                        try:
                            render_weather(arrive, arrive_date)
                        except Exception as e:
                            st.error(f"날씨 조회 중 오류: {e}")
                            st.caption("weather_module.py가 같은 폴더에 있는지, 인터넷 연결/라이브러리 설치를 확인하세요.")
                

                except Exception as e:
                    st.error(f"오류: {e}")
                    
    if page == "챗봇":
        df = pd.read_excel('FNQ.xlsx')
        docs = df_to_documents(df, use_cols=["질문", "답변", "분류"])
        st.session_state.vs = build_vectorstore(docs)

        if "vs" in st.session_state:
            q = st.text_input("질문을 입력하세요")
            context, src_docs = retrieve_context(q, st.session_state.vs, k=15)
            answer = ask_llm_with_context(q, context)
            st.subheader("답변")
            st.write(answer)

            st.subheader("근거 (상위 문서)")
            for i, d in enumerate(src_docs, 1):
                st.markdown(f"**#{i}** meta: `{d.metadata}`")
                st.code(d.page_content[:1000])


if __name__ == "__main__":
    main()