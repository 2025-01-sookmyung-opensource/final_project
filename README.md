# 🍱 당뇨병 환우를 위한 개인 맞춤형 식단 관리 서비스 🍱
<img width="1512" alt="image" src="https://github.com/user-attachments/assets/404fc008-39fc-46de-9a99-5620d06b102e" />

## [목차]

- [1. 프로젝트 소개]()
  - [1.1 주요 기능]()
  - [1.2 실행 예시 결과]()
- [2. 개발 기간]()
- [3. 기술 스택]()
  - [3.1 개발 환경 & 툴]()
  - [3.2 언어 & 프레임워크]()
  - [3.3 배포]()
- [4. 기술 아키텍처]()
  - [4.1 flowchart]()
  - [4.2 서비스 흐름]()
- [5. 사용한 데이터 셋 & 모듈]()
  - [5.1 AIhub :: 음식 이미지 및 영양정보 텍스트]()
  - [5.2 음식 분류 데이터셋 라벨링 프로젝트]()
  - [5.3 YOLOv8n 학습설정]()
  - [ 5.4 양추정 :: MobileNetV3 & EfficientNet]()
  - [5.5 필수 라이브러리 설치]()
- [6. License]()
- [7. 팀원 소개]()


## 1. 프로젝트 소개
본 프로젝트는 **당뇨병 환우의 식단 이미지로부터 음식의 종류와 양을 분류**하여 영양성분을 분석하고 개인 맞춤형 피드백을 제공하는 서비스입니다.

- 배포 사이트
    - [YOLOv8 & MobileNetV3 배포 사이트](https://streamlit-nutrition-app-436215849351.asia-northeast3.run.app/)
    - [YOLOv8 & EfficientNet 배포 사이트](https://streamlit-nutrition-app-k7wpx6hjma-du.a.run.app/)

- 예상 기대 효과
  - 단순 기록을 넘어, 실질적인 식단 피드백과 자가관리 지원이 가능한 서비스로 확장하는 것을 목표로 함.
  - 모바일 및 웹 환경(Colab&Streamlit)에서도 동작 가능한 경량 딥러닝 모델을 구축하고, 사용자 정보 기반의 혈당 예측 회귀 모델을 함께 개발함.
  
### 1.1 주요 기능
1. 식단 이미지의 음식 인식 후 분류 (YoloV8)
2. 인식한 음식의 양 추정(MobileNetV3, EfficientNet)
3. 추정한 데이터들을 바탕으로 한 영양정보DB로 streamlit에서 결과분석 및 시각화
4. Gemini API를 이용한 식단 피드백 제공

### 1.2 실행 예시 결과
- 시연영상
  
[![구현 동영상 썸네일](https://img.youtube.com/vi/jqrcQ8ywoS4/0.jpg)](https://www.youtube.com/watch?v=jqrcQ8ywoS4)


## 2. 개발 기간 
2025년 5월 13일 ~ 2025년 6월 22일

## 3. 기술 스택
### 3.1 개발 환경 & 툴
![Windows](https://img.shields.io/badge/Windows-0078D6?style=flat&logo=windows&logoColor=white)
![macOS](https://img.shields.io/badge/macOS-000000?style=flat&logo=apple&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white)
![VSCode](https://img.shields.io/badge/VSCode-007ACC?style=flat&logo=visualstudiocode&logoColor=white)
![Colab](https://img.shields.io/badge/Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)
![Roboflow](https://img.shields.io/badge/Roboflow-10172A?style=flat&logo=roboflow&logoColor=white)

### 3.2 언어 & 프레임워크
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

### 3.3 배포
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![GCP](https://img.shields.io/badge/Google_Cloud-4285F4?style=flat&logo=googlecloud&logoColor=white)


## 4. 기술 아키텍처
## 4.1 flowchart
![image](https://github.com/user-attachments/assets/a9367596-c3aa-4942-91cd-f1c9daee0f1c)

## 4.2 서비스 흐름
![image](https://github.com/user-attachments/assets/b3843a83-c7c7-404c-a314-7aef1fd685a8)


## 5 사용한 데이터 셋 & 모듈
### 5.1 [AIhub :: 음식 이미지 및 영양정보 텍스트](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=74)
    - 음식분류 AI 데이터 영양DB.xlsx
    - 양추정 : 양추정_이미지_VAL_0422_add.zip l 5.49 GB 
    - 음식분류 : 음식분류_이미지_TRAIN_001_1223_add.zip l 27.44 GB
### 5.2 음식 분류 데이터셋 라벨링 프로젝트
- 데이터 라벨링 프로젝트 : [Roboflow :: opensource_final_yoon](https://app.roboflow.com/yoon-pvmwt/opensource_final_yoon/models)
- 음식 클래스 수: **21종**
- 각 클래스 당 100장의 이미지르 Roboflow를 통해 수집 및 구성함
<details>
<summary><strong>클래스별 이미지 수</strong></summary>
<br>

|  Class Name         |  이미지 수 |
|------------------------|---------------|
| bakedpotato            | 103           |
| bibimbap (비빔밥)         | 112           |
| chickensteak           | 100           |
| coldsoymilknoodles     | 100           |
| crabstick              | 100           |
| eeldonburi             | 100           |
| grilledmackerel        | 100           |
| grilledribs            | 100           |
| japchae                | 100           |
| jjajangmyeon           | 100           |
| kimbap                 | 101           |
| kimchifriedrice        | 99            |
| kimchistew             | 97            |
| mungbeansprouts        | 98            |
| noodles                | 100           |
| pasta                  | 100           |
| roastedsweetpotato     | 100           |
| scrambledegg           | 100           |
| spinach                | 100           |
| steamerice             | 100           |
| tunasandwich           | 100           |

</details>

### 5.3 음식 분류 :: YOLOv8n 학습설정
- epoch 수 : 100
- 이미지 크기 (전처리) : 640x640
- batch 크기 : 16
- 학습 프레임 워크 → ![Colab](https://img.shields.io/badge/Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white) + T4 GPU 환경
- [우진 :: Image_Train_v8.ipynb](https://github.com/2025-01-sookmyung-opensource/final_project/blob/woojin/Image_Train_v8.ipynb)
- [지윤 :: modelTrain ](https://github.com/2025-01-sookmyung-opensource/final_project/tree/pre-yoon/modelTrain)
  
### 5.4 양추정 :: MobileNetV3 & EfficientNet
- 클래스 구조
```
김밥/
├── Q1 / (아주 적음)
├── Q2 /
├── Q3 /
├── Q4 /
└── Q5/ (매우 많음)
```
- EfficientNet 모델 학습
  - 이미지 디렉터리, txt 디렉터리 구조 및 YOLO 형식 바운딩 박스 정보는 이전과 동일
  - 이미지 크기 (전처리) : 224 x 224 → Tensor 변환
  - 데이터 증강 : RandomResizedCrop 후 Resize로 크기 맞추기 전략
  - 모델의 전체 정확도는 약 30.4%로, 무작위 추정보다 크게 높지 않아 분류 성능이 전반적으로 낮음
- MobileNetV3 모델 학습
  - 사전학습된 MobileNetV3-Small 모델을 불러온 뒤, 최종 분류 레이어를 5개의 섭취량 등급(Q1~Q5)으로 수정
  - 이미지 크기 (전처리) : 224 x 224 → Tensor 변환
  - epoch : 10회
- [민지 :: MobileNetV3_amount_estimator.ipynb ](https://github.com/2025-01-sookmyung-opensource/final_project/blob/pre-min/MobileNetV3_amount_estimator.ipynb)
- [효정 :: EfficientNet_amount_estimator.ipynb](https://github.com/2025-01-sookmyung-opensource/final_project/blob/amount-estimation-model/efficientnet_amount_estimator.ipynb)

### 5.5 필수 라이브러리 설치

```bash
# PyTorch, torchvision, timm (MobileNetV3, EfficientNet 포함)
pip install torch torchvision timm

# YOLOv8 (ultralytics)
pip install ultralytics

```
- Google Colab (T4 GPU)
- 또는 로컬 GPU (CUDA 지원)

## 6. License
이 서비스는 다음의 오픈소스 AI 모델을 포함합니다. 

<details>
<summary><strong><span style="font-weight:bold">YOLOv8 (Ultralytics): GPLv3 License</span></strong></summary>

본 프로젝트는 Ultralytics의 <a href="https://github.com/ultralytics/ultralytics">YOLOv8</a>을 기반으로 객체 탐지를 수행합니다.  
YOLOv8은 <strong>GNU General Public License v3.0 (GPLv3)</strong>을 따르며, 이 라이선스를 존중합니다.

<blockquote>
※ 해당 라이선스 조건에 따라, YOLOv8을 사용하는 모든 소스코드는 공개되어야 하며, 파생 프로젝트 역시 GPLv3을 따라야 할 수 있습니다.
</blockquote>

</details>

<details>
<summary><strong><span style="font-weight:bold">EfficientNet (Google): Apache License 2.0</span></strong></summary>

본 프로젝트는 Google의 <a href="https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet">EfficientNet</a> 아키텍처를 기반으로 양 추정 모델 중 하나를 구현하였습니다.  
EfficientNet은 <strong>Apache License 2.0</strong>을 따릅니다.

<blockquote>
※ Apache 2.0 라이선스는 상업적 이용 및 수정이 가능합니다.(단, 라이선스 고지를 유지)
</blockquote>

</details>

<details>
<summary><strong><span style="font-weight:bold">MobileNetV3 (Google): Apache License 2.0</span></strong></summary>

본 프로젝트는 Google의 <a href="https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v3.md">MobileNetV3</a> 아키텍처를 기반으로 양 추정 모델 중 하나를 구현하였습니다.  
MobileNetV3 역시 <strong>Apache License 2.0</strong>을 따릅니다.

<blockquote>
※ Apache 2.0 라이선스는 상업적 사용과 수정이 자유롭고, 소스코드 공개 의무는 없습니다. (단, 라이선스 고지를 유지)
</blockquote>

</details>




## 7. 팀원 소개 

| <img src="https://github.com/dreaminji99.png?size=80" width="80"/> | <img src="https://github.com/rb37lu71.png?size=80" width="80"/> | <img src="https://github.com/Yoon0221.png?size=80" width="80"/> | <img src="https://github.com/woojin-devv.png?size=80" width="80"/> |
|---|---|---|---|
| **구민지** | **김효정** | **신지윤** | **최우진** |
| Leader / 양추정 모델 (MobileNetV3) | 양추정 모델 (EfficientNet) | 음식 분류 모델(YoloV8)| 음식 분류 모델(YoloV8) |
| [@dreaminji99](https://github.com/dreaminji99) | [@rb37lu71](https://github.com/rb37lu71) | [@Yoon0221](https://github.com/Yoon0221) | [@woojin-devv](https://github.com/woojin-devv) |

  
