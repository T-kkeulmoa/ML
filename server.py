from flask import Flask, request, Response
from PIL import Image
import io
import torch
import json
from requests_toolbelt import MultipartEncoder,MultipartDecoder
import os

app = Flask(__name__)

@app.route('/api/v1/trash', methods=['POST'])
def process_image():
    if request.method == 'POST':
        # YOLOv5 모델 로드
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')

        # 멀티파트 이미지 파일 수신
        image_file = request.files['multipartFile']
        image = Image.open(image_file.stream)

        # 이미지 처리 (YOLOv5 추론)
        results = model(image)
        results.save(filename)

        # 결과에서 레이블 추출
        labels = results.xyxy[0][:, -1].cpu().numpy()
        label_names = list(set([results.names[int(label)] for label in labels]))
        
        filename = './result/result.jpg'
        
        body = MultipartEncoder(
            fields={
                "code": 200,
                "message": "다신 안해 병신짓",
                "trashAnalyzeImage": (os.path.basename(filename), open(filename, 'rb'), 'image'),
                "trashInfoList": label_names
            }
        )

        return Response(body.to_string(), mimetype=body.content_type)

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
