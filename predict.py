import torch

# 훈련된 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')

# 테스트 이미지 로드
img = 'result/now.jpg'

# 추론 수행
results = model(img)

# 결과 출력
results.print()  # 추론 결과 출력
results.show()   # 결과 이미지 표시
results.save()   # 결과 이미지 저장

# 결과에서 레이블, 신뢰도, 바운딩 박스 좌표 추출
labels = results.xyxy[0][:, -1].cpu().numpy()  # 레이블
# 레이블 이름 얻기
label_names = list(set([results.names[int(label)] for label in labels]))

print(label_names)
