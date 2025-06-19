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
os.environ['PORT'] = '8080'

# ───────────────────────── 모델 로딩 ─────────────────────────
@st.cache_resource
def load_yolo_model():
    return YOLO('best.pt')

@st.cache_resource
def load_portion_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    state_dict = torch.load('portion_model.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ───────────────────────── 데이터 로딩 ─────────────────────────
@st.cache_data
def load_nutrition_data():
    df = pd.read_csv("nutrition_db.csv")
    df['음식명_lower'] = df['음식명'].str.strip().str.lower()  # 한 번만 생성
    return df

nutrition_df = load_nutrition_data()

# ───────────────────────── 유틸 함수 ─────────────────────────
def preprocess_portion_image(image_np):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image_np).unsqueeze(0)

def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 1)

def get_bmi_category(bmi):
    if bmi < 18.5: return "저체중"
    elif bmi < 23: return "정상"
    elif bmi < 25: return "과체중"
    else: return "비만"

def estimate_caloric_needs(age, gender, height_cm, weight_kg):
    if gender == "남성":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    elif gender == "여성":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age
    return round(bmr * 1.55)

# ───────────────────────── UI 구성 ─────────────────────────
st.set_page_config(page_title='🍱 식단 이미지 분석기', layout='wide')
st.markdown("""
    <h1 style='text-align: center;'>🍱 식단 이미지 분석기</h1>
    <p style='text-align: center;'>YOLOv8으로 음식 탐지, ResNet50으로 섭취량 예측, 영양정보 + 개인 건강 분석까지!</p>
""", unsafe_allow_html=True)

# ───────────────────────── 사이드바 입력 ─────────────────────────
st.sidebar.header("🤖 사용자 정보 입력")
age = st.sidebar.number_input("나이", min_value=0, max_value=120, value=25)
gender = st.sidebar.selectbox("성별", ["남성", "여성", "기타"])
height = st.sidebar.number_input("키(cm)", min_value=50, max_value=250, value=170)
weight = st.sidebar.number_input("몸무게(kg)", min_value=20, max_value=200, value=60)

bmi = calculate_bmi(weight, height)
bmi_status = get_bmi_category(bmi)
recommended_kcal = estimate_caloric_needs(age, gender, height, weight)

st.sidebar.markdown(f"""
✨ **개인 건강 정보**
- BMI: **{bmi}** ({bmi_status})
- 일일 권장 칼로리: **{recommended_kcal} kcal**
""")

st.sidebar.header("📤 이미지 업로드")
uploaded = st.sidebar.file_uploader('이미지를 업로드하세요', ['jpg', 'jpeg', 'png'])

model = load_yolo_model()
portion_model = load_portion_model()

# ───────────────────────── 이미지 처리 ─────────────────────────
if uploaded:
    pil_img = Image.open(uploaded).convert('RGB')
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    st.image(rgb, caption='🖼️ 업로드한 이미지', use_container_width=True)

    res = model.predict(bgr, conf=0.25, verbose=False)[0]
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    boxes = res.boxes.xyxy.cpu().numpy()
    names = model.names

    st.subheader('🍽️ 인식된 항목')
    detected = {}
    for cid in cls_ids:
        name = names[cid]
        detected[name] = detected.get(name, 0) + 1

    if detected:
        for name, count in detected.items():
            st.write(f"• {name}: {count}개")
    else:
        st.warning('❗ 아무것도 인식되지 않았습니다.')

    st.image(res.plot(), caption='YOLO 탐지 결과', use_container_width=True)

    # ───────────── 객체별 분석 ─────────────
    st.subheader('🔍 객체별 섭취량 + 영양정보')

    for i, (cid, box) in enumerate(zip(cls_ids, boxes), start=1):
        x1, y1, x2, y2 = map(int, box)
        crop_rgb = rgb[y1:y2, x1:x2]
        if crop_rgb.size == 0:
            continue

        label = names[cid].strip().lower().replace(" ", "")
        matched = nutrition_df[nutrition_df['음식명_lower'] == label]
        if matched.empty:
            st.warning(f"⚠️ '{label}'에 해당하는 영양정보가 없습니다. 유사 항목을 확인하세요.")
            similar = nutrition_df[nutrition_df['음식명_lower'].str.contains(label[:4])]
            st.dataframe(similar[['음식명', '한글명']])

        kor_name = matched['한글명'].values[0] if not matched.empty else label

        st.markdown(f"### 🍛 {i}. {kor_name}")
        cols = st.columns([1, 2])
        with cols[0]:
            st.image(crop_rgb, width=250)
        with cols[1]:
            try:
                input_tensor = preprocess_portion_image(crop_rgb)
                with torch.no_grad():
                    output = portion_model(input_tensor)
                    portion_value = output.item()
                st.success(f'🥄 섭취량 점수: **{portion_value:.2f}**')

                if not matched.empty:
                    nutrition_info = matched.iloc[0][1:-2]  # '음식명_lower', '한글명' 제외
                    info_df = pd.DataFrame(nutrition_info).reset_index()
                    info_df.columns = ['항목', '값']
                    st.dataframe(info_df, use_container_width=True, height=300)
                else:
                    st.info('⚠️ 등록된 영양정보가 없습니다.')
            except Exception as e:
                st.warning(f'⚠️ 분석 중 오류 발생: {e}')

        st.markdown("---")
