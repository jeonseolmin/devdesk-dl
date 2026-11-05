# DevDesk 딥러닝 과제

개발자 데스크톱 앱 DevDesk에 붙이려고 만든 딥러닝 모델 네 가지의 코드, 리포트, 실행 페이지입니다.
앱 본체는 이 저장소에 없습니다.

| 과제 | 모델 | 비교 대상 |
|---|---|---|
| 웨이크워드 "자비스" | CNN (log-mel) | 전이학습 (openWakeWord 임베딩 + 분류기) |
| 다음에 쓸 앱 예측 | GRU | 되돌아가기, 마르코프, LightGBM |
| 하루 패턴 이상 탐지 | 조건부 오토인코더 | 평균과의 거리, PCA |
| 집중 이탈 예측 | GRU | 사람별 평균, 직전 10분처럼, LightGBM |

결과 요약은 [ml/reports/presentation.md](ml/reports/presentation.md)에 있습니다.

## 실행

```sh
pip install -r ml/requirements.txt
streamlit run ml/streamlit_app.py
```

첫 탭의 점검표에서 모델마다 동작 여부를 확인하고, 과제별 탭에서 직접 입력해 볼 수 있습니다.

## 저장소에 없는 것

- **웨이크워드 모델 `wake.onnx`:** 개인 목소리로 학습한 모델이라 넣지 않았습니다. 이 모델이 없으면 "자비스" 패널만 꺼지고 나머지는 그대로 동작합니다.
- **학습 데이터:** 원본 데이터는 넣지 않았습니다.

## 데이터와 라이선스

- 활동 모델 3개는 [BEHACOM](https://doi.org/10.17632/cg4br62535.2)(CC BY 4.0)으로 학습했습니다. `ml/models/demos.json`에는 학습에 쓰지 않은 사용자별 하루 패턴 집계값이 들어 있습니다.
- 웨이크워드 학습에는 XTTS-v2(CPML, 비상업)로 합성한 음성과 Speech Commands를 썼습니다.
