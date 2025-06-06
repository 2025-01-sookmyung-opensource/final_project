class_map = {
    "백도": "whitepeach",
    "방울토마토": "cherrytomato",
    "밥": "rice",
    "먹는 밤": "chestnut",
    "바베큐치킨": "bbqchicken",
    "바닐라아이스크림": "vanillaicecream",
    "바나나": "banana",
    "바게트빵": "baguette"
}

import os
from pathlib import Path
from collections import defaultdict

# 클래스 한글 → 영어 맵핑
class_map = {
    "백도": "whitepeach",
    "방울토마토": "cherrytomato",
    "밥": "rice",
    "먹는 밤": "chestnut",
    "바베큐치킨": "bbqchicken",
    "바닐라아이스크림": "vanillaicecream",
    "바나나": "banana",
    "바게트빵": "baguette"
}

# 경로 설정
base_image_dir = Path(r"C:\Users\user\Downloads\건강관리를 위한 음식 이미지\Training\yoon_data")
base_label_dir = Path(r"C:\Users\user\Downloads\건강관리를 위한 음식 이미지\Training\yoon_data_label")

# 한글 클래스명을 이름순 정렬
sorted_korean_classes = sorted(class_map.keys())

# 번호 매기기 위한 딕셔너리 초기화
file_counters = defaultdict(int)

# 이미지 파일명 변경
for class_kor in sorted_korean_classes:
    class_eng = class_map[class_kor]
    
    # 이미지 폴더 안 파일 목록
    image_folder = base_image_dir / class_kor
    if not image_folder.exists():
        continue
    
    for file in sorted(image_folder.glob("*.jpg")):
        file_counters[class_kor] += 1
        new_name = f"{class_eng}_{file_counters[class_kor]:03d}.jpg"
        new_path = file.with_name(new_name)
        print(f"Renaming: {file.name} -> {new_name}")
        file.rename(new_path)

# 라벨 파일명 변경
file_counters.clear()  # 번호 초기화
for class_kor in sorted_korean_classes:
    class_eng = class_map[class_kor]
    
    label_folder = base_label_dir / class_kor
    if not label_folder.exists():
        continue
    
    for file in sorted(label_folder.glob("*.json")):
        file_counters[class_kor] += 1
        new_name = f"{class_eng}_{file_counters[class_kor]:03d}.json"
        new_path = file.with_name(new_name)
        print(f"Renaming: {file.name} -> {new_name}")
        file.rename(new_path)
