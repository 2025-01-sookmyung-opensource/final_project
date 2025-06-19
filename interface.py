import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os

# ───────────────────────── 환경 설정 ─────────────────────────
# Streamlit이 포트 8080에서 실행되도록 설정
os.environ['PORT'] = '8080'
# ───────────────────────── 모델 로딩 ─────────────────────────
@st.cache_resource
def load_yolo_model():
    return YOLO('best.pt')

@st.cache_resource
def load_portion_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)  # 회귀 또는 binary/1-class 분류라면 출력 1
    state_dict = torch.load('portion_model.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ───────────────────────── 데이터 로딩 ─────────────────────────
@st.cache_data
def load_nutrition_data():
    return pd.read_csv("nutrition_db.csv")

nutrition_df = load_nutrition_data()

# 전처리
def preprocess_portion_image(image_np):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image_np).unsqueeze(0)

# ───────────────────────── Streamlit UI ─────────────────────────
st.set_page_config(page_title='🍱 식단 이미지 분석기', layout='centered')
st.title('🍱 식단 이미지 분석기')
st.markdown('업로드한 음식 사진에서 **YOLOv8**으로 음식을 탐지하고, '
            '**ResNet50**으로 섭취량을 예측하며, '
            '매칭된 **영양정보**를 함께 제공합니다.')

model = load_yolo_model()
portion_model = load_portion_model()

uploaded = st.file_uploader('📤 이미지를 업로드하세요', ['jpg', 'jpeg', 'png'])

if uploaded:
    pil_img = Image.open(uploaded).convert('RGB')
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    st.image(rgb, caption='🖼️ 업로드한 이미지', use_container_width=True)

    # ───────────── YOLO 탐지 ─────────────
    res = model.predict(bgr, conf=0.25, verbose=False)[0]
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    boxes = res.boxes.xyxy.cpu().numpy()
    names = model.names

    detected = {}
    for cid in cls_ids:
        name = names[cid]
        detected[name] = detected.get(name, 0) + 1

    st.subheader('🍽️ 인식된 항목')
    if detected:
        for name, count in detected.items():
            st.write(f'• {name}: {count}개')
    else:
        st.warning('❗ 아무것도 인식되지 않았습니다.')

    st.image(res.plot(), caption='YOLO 탐지 결과', use_container_width=True)

    if st.download_button(
        label='📥 결과 이미지 다운로드',
        data=cv2.imencode('.jpg', res.plot())[1].tobytes(),
        file_name='result.jpg',
        mime='image/jpeg'
    ):
        st.success('이미지를 저장했습니다!')

    # ───────────── 객체별 분석 ─────────────
    st.subheader('🔍 객체별 섭취량 + 영양정보')

    for i, (cid, box) in enumerate(zip(cls_ids, boxes), start=1):
        x1, y1, x2, y2 = map(int, box)
        crop_rgb = rgb[y1:y2, x1:x2]

        if crop_rgb.size == 0:
            continue

        label = names[cid]
        st.image(crop_rgb, caption=f'{i}. {label}', width=240)

        try:
            # 섭취량 예측
            input_tensor = preprocess_portion_image(crop_rgb)
            with torch.no_grad():
                output = portion_model(input_tensor)
                value = output.item()
            st.success(f'🍽️ 섭취량 점수(회귀 출력): **{value:.2f}**')

            # 영양정보 출력
            matched = nutrition_df[nutrition_df['음식명'].str.lower() == label.lower()]
            if not matched.empty:
                st.markdown('📊 **영양 정보 (1회 제공량 기준)**')
                
                nutrition_info = matched.iloc[0][1:]  # 음식명 제외
                info_df = pd.DataFrame(nutrition_info).reset_index()
                info_df.columns = ['항목', '값']
                st.table(info_df)  # 또는 st.dataframe(info_df, use_container_width=True)
            else:
                st.info('⚠️ 등록된 영양정보가 없습니다.')


            # YOLO 재탐지 (선택적)
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            sub = model.predict(crop_bgr, conf=0.25, verbose=False)[0]
            sub_ids = sub.boxes.cls.cpu().numpy().astype(int)
            sub_det = {}
            for sid in sub_ids:
                sname = names[sid]
                sub_det[sname] = sub_det.get(sname, 0) + 1

            if sub_det:
                st.markdown(f'**{i}. 내부 구성 분석**')
                for sname, count in sub_det.items():
                    st.write(f'↳ {sname}: {count}개')

            st.image(sub.plot(), caption=f'{i}. 상세 분석 결과', width=300)

        except Exception as e:
            st.warning(f'⚠️ 분석 중 오류 발생: {e}')
