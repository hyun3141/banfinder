import streamlit as st
import requests

st.title("YOLO 이미지 분석 API 테스트")

uploaded_file = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    files = {"file": uploaded_file.getvalue()}
    #response = requests.post("http://localhost:8081/api_test/process-image2")
    response = requests.post("http://localhost:8081/api/process-image", files={"file": uploaded_file})
    if response.status_code == 200:
        res_json = response.json()
        st.write(res_json)
        # st.write(res_json["result_gpt_desc"])
        # st.image(res_json["result_yolo_image"], caption="YOLO 분석 결과")
        # st.write(res_json["result_yolo_desc"])
    else:
        st.error("API 호출 실패")