import torch
from PIL import Image, ImageDraw, ImageFont
import random
import base64
from io import BytesIO
import argparse

def main(args):
    # 한글 레이블 매핑
    label_mapping = {
        "Cup": "컵",
        "Carton": "종이박스",
        "Styrofoam piece": "스티로폼",
        "Plastic glooves": "비닐 장갑",
        "Broken glass": "깨진 유리",
        "Battery": "배터리",
        "Straw": "빨대",
        "Blister pack": "약 껍질",
        "Plastic bag & wrapper": "플라스틱 포장지",
        "Paper": "종이",
        "Can": "캔",
        "Lid": "뚜껑",
        "Plastic container": "플라스틱 용기",
        "Plastic utensils": "플라스틱 식기",
        "Bottle cap": "병 뚜껑",
        "Paper bag": "종이 가방",
        "Rope & strings": "로프 및 끈",
        "Food waste": "음식물 쓰레기",
        "Scrap metal": "고철",
        "Bottle": "병",
        "Glass jar": "유리잔",
        "Squeezable tube": "짜는 튜브",
        "Aluminium foil": "알루미늄 호일",
        "Clear plastic bottle": "투명 플라스틱 병",
        "Other plastic bottle": "기타 플라스틱 병",
        "Glass bottle": "유리병"
    }


    # 훈련된 모델 로드
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')


    # 테스트 이미지 로드
    img_path = 'result/now.jpg'
    img = Image.open(BytesIO(base64.b64decode(base64_string)))

    # 추론 수행
    results = model(img_path)

    # 결과에서 바운딩 박스 좌표 추출
    boxes = results.xyxy[0][:, :4].cpu().numpy()  # xmin, ymin, xmax, ymax 좌표
    labels = results.xyxy[0][:, -1].cpu().numpy()  # 레이블

    # 클래스별 색상 설정
    class_colors = {label: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for label in set(labels)}

    # 이미지를 복사하여 그리기 시작
    draw = ImageDraw.Draw(img)

    # 폰트 설정
    try:
        font = ImageFont.truetype(font = "./font/SpoqaHanSansNeo-Regular.otf",size= 70)
    except IOError:
        font = ImageFont.load_default()

    # 바운딩 박스
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        color = class_colors[label]
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=10)  
        label_text = label_mapping[results.names[int(label)]]
        text_bbox = draw.textbbox((xmin, ymin), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_background = [xmin, ymin - text_height, xmin + text_width, ymin]
        draw.rectangle(text_background, fill=color)
        draw.text((xmin, ymin - text_height), label_text, fill="black", font=font)

    img.show()

    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # 레이블 이름 얻기

    label_names = list(set([label_mapping[results.names[int(label)]] for label in labels]))


    print(img_str,label_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base64_code",type=str)

    args = parser.parse_args()
    main(args.base64_code)