# Prototype
prototype for Task-Oriented Semantic Communication System with Time-Series Data 

preprocess.py : 데이터 로딩 및 전처리 (CSV → 시계열 window + 상태 지표)
model.py : Semantic Encoder-Decoder 모델 정의
train.py : 모델 학습 루프 (loss 계산 포함)
main.py : 전체 파이프라인 실행 (학습 + 평가)
performance.py : 성능 평가 및 시각화 (복원 정확도 + 상태 예측 성능)
utils.py : 스케일링, 시각화 등 유틸 함수 정의