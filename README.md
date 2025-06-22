# 🍱 당뇨병 환우를 위한 개인 맞춤형 식단 관리 서비스 🍱


## 프로젝트소개
본 프로젝트는 **당뇨병 환우의 식단 이미지로부터 음식의 종류와 양을 분류**하여 영양성분을 분석하고 개인 맞춤형 피드백을 제공하는 서비스.

- 배포 사이트
    - [YOLOv8 & MobileNetV3 배포 사이트](https://streamlit-nutrition-app-436215849351.asia-northeast3.run.app/)
    - [YOLOv8 & EfficientNet 배포 사이트](https://streamlit-nutrition-app-k7wpx6hjma-du.a.run.app/)

## 📅 개발 기간
2025년 5월 13일 ~ 2025년 6월 22일

### 🖥️ 개발 환경 & 툴
![Windows](https://img.shields.io/badge/Windows-0078D6?style=flat&logo=windows&logoColor=white)
![macOS](https://img.shields.io/badge/macOS-000000?style=flat&logo=apple&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white)
![VSCode](https://img.shields.io/badge/VSCode-007ACC?style=flat&logo=visualstudiocode&logoColor=white)
![Colab](https://img.shields.io/badge/Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)
![Roboflow](https://img.shields.io/badge/Roboflow-10172A?style=flat&logo=roboflow&logoColor=white)

### 💻 언어 & 프레임워크
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

### 🚀 배포
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![GCP](https://img.shields.io/badge/Google_Cloud-4285F4?style=flat&logo=googlecloud&logoColor=white)



## 사용한 데이터 셋
### [AIhub :: 음식 이미지 및 영양정보 텍스트](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=74)
    - 음식분류 AI 데이터 영양DB.xlsx
    - 양추정 : 양추정_이미지_VAL_0422_add.zip l 5.49 GB 
    - 음식분류 : 음식분류_이미지_TRAIN_001_1223_add.zip l 27.44 GB
### 🍽️ 음식 분류 데이터셋 라벨링

- 데이터 라벨링 프로젝트 : [Roboflow :: opensource_final_yoon](https://app.roboflow.com/yoon-pvmwt/opensource_final_yoon/models)
- 음식 클래스 수: **21종**

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


## 📢 OSS NOTICE
이 서비스는 다음의 오픈소스 AI 모델을 포함합니다. 

<details>
<summary><strong>YOLOv8 (Ultralytics): GPLv3 License </strong></summary>
 
  본 프로젝트는 Ultralytics의 [YOLOv8](https://github.com/ultralytics/ultralytics)을 기반으로 객체 탐지를 수행합니다.  
  YOLOv8은 **GNU General Public License v3.0 (GPLv3)**을 따르며, 이 라이선스를 존중합니다.

> ※ 해당 라이선스 조건에 따라, YOLOv8을 사용하는 모든 소스코드는 공개되어야 하며, 파생 프로젝트 역시 GPLv3을 따라야 할 수 있습니다.

</details>



### ✅ 필수 라이브러리 설치

```bash
# PyTorch, torchvision, timm (MobileNetV3, EfficientNet 포함)
pip install torch torchvision timm

# YOLOv8 (ultralytics)
pip install ultralytics

```

## 💻 권장 환경
Google Colab (T4 GPU)
또는 로컬 GPU (CUDA 지원)

## 📜 flowchart

![alt text](흐름요약도-1.png)

## 서비스 흐름
![image](https://github.com/user-attachments/assets/b3843a83-c7c7-404c-a314-7aef1fd685a8)


##  ✔ 주요 기능
1. 식단 이미지의 음식 인식 후 분류 (YoloV8)
2. 인식한 음식의 양 추정(MobileNetV3, EfficientNet)
3. 추정한 데이터들을 바탕으로 한 영양정보DB로 streamlit에서 결과분석 및 시각화
4. Gemini API를 이용한 식단 피드백 제공

##  ✔ 실행 예시 결과
- 구현 동영상
>https://www.youtube.com/watch?v=jqrcQ8ywoS4&feature=youtu.be

##   팀원 소개 

| <img src="https://github.com/dreaminji99.png?size=80" width="80"/> | <img src="https://github.com/rb37lu71.png?size=80" width="80"/> | <img src="https://github.com/Yoon0221.png?size=80" width="80"/> | <img src="https://github.com/woojin-devv.png?size=80" width="80"/> |
|---|---|---|---|
| **구민지** | **김효정** | **신지윤** | **최우진** |
| Leader / 양추정 모델 (MobileNetV3) | 양추정 모델 (EfficientNet) | 음식 분류 모델(YoloV8)| 음식 분류 모델(YoloV8) |
| [@dreaminji99](https://github.com/dreaminji99) | [@rb37lu71](https://github.com/rb37lu71) | [@Yoon0221](https://github.com/Yoon0221) | [@woojin-devv](https://github.com/woojin-devv) |

  
