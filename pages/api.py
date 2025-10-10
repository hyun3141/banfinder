import streamlit as st
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from banfinder_openai_def import *
from ultralytics import YOLO
import io


app = FastAPI()

@app.post("/api/process-image")
async def main(file: UploadFile = File(...)):
    st.title("빈 페이지")
    st.write("이 페이지는 내용이 없습니다.")
    
    img_bytes = await file.read()
    img_stream = io.BytesIO(img_bytes)
    img = Image.open(img_stream).convert("RGB")

    
    result_gpt = classify_items(img)  # 당신이 이미 가진 함수 사용
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
    desc_str = ''.join(str(i) for i in desc)
    
    # OpenCV 이미지(BGR)를 RGB로 변환해서 스트림릿에 출력
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    st.image(result_img_rgb, caption="분석 결과", use_container_width=True)


    # 3. 결과 이미지 PNG -> base64 data-uri 변환
    import base64

    pil_img = Image.fromarray(result_img_rgb)  # numpy array -> PIL Image 변환

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")  # PIL 이미지 저장

    b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64_img}"

    # 4. JSON 형태로 응답
    return JSONResponse(content={
        "result_gpt_desc": result_gpt,
        "result_yolo_image": data_uri,
        "result_yolo_desc": desc_str,
    })

if __name__ == "__main__":
    main()