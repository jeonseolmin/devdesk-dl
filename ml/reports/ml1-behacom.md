# ML1 집중 상태 분류기 — BEHACOM 규칙 라벨 결과 (2026-10-26)

재현: `uv run --group ml python -m ml.train_focus` (약 1분). 전체 지표는 `ml/models/focus_lgbm_metrics.json`(git 제외).
학습 노트북: `ml/notebooks/ml1_focus_behacom.ipynb` (같은 함수로 단계별 확인).
**성격: 파이프라인 검증.** 라벨이 규칙이므로 점수는 "규칙을 복원하는가"이지 실제 집중을 맞히는가가 아님(아래 해석).

- 데이터: BEHACOM 12명 중 라벨 행이 있는 11명, 라벨 행 22,112 = break 18,619 / focus 2,333 / distracted 1,160.
  유휴 1시간 초과는 "자리 비움"으로 라벨만 NaN(`focus_features.UNATTENDED_SEC`, 행은 유지); 이 처리 전엔 break가 96%였음.
  행 간격 10분 이상은 세션 경계(PC 꺼짐)로 유휴 계산을 끊음(`behacom.CONTIGUOUS_SEC`).
- 검증: leave-one-user-out(GroupKFold, 사용자 11개 fold), out-of-fold 예측을 모아서 계산.
- 특징: 이전 5/15분 범주 비율·전환 평균·유휴 평균·활동 비율, 직전 1분 범주/유휴, 시각(sin/cos)·요일. 현재 분은 제외.

| 모델 | macro-F1 | F1 focus/distracted만 | F1 focus | F1 distracted | F1 break |
|---|---|---|---|---|---|
| majority | 0.305 | 0.000 | 0.000 | 0.000 | 0.914 |
| persistence(직전 분 규칙 그대로) | 0.890 | 0.849 | 0.738 | 0.961 | 0.972 |
| logreg | 0.958 | 0.940 | 0.979 | 0.900 | 0.994 |
| random_forest | 0.987 | 0.981 | 0.979 | 0.983 | 0.998 |
| **lightgbm (배포용)** | **0.986** | **0.979** | 0.983 | 0.976 | 0.999 |
| lightgbm, 랜덤 분할(누수 비교) | 0.990 | 0.986 | 0.988 | 0.984 | 0.999 |

- LightGBM 상위 특징(gain): active_ratio_5m 0.29, prev_idle_sec 0.23, ratio_game_15m 0.23, idle_mean_5m 0.12, ratio_game_5m 0.06.
- 모델 파일 917 KB(`focus_lgbm.txt`), 1행 예측 약 1 ms → 앱에서 1분마다 돌리기에 충분.
- 배포 모델과 LightGBM 외 학습기 중 macro-F1 최고 모델의 배포 비용(`train_focus.deploy_cost`, `focus_lgbm_metrics.json`의 `deploy_cost`, 2026-10-27 측정).
  조건: 전체 데이터로 학습한 최종 모델, 단일 스레드(`n_jobs=1`, 앱은 프로세스 안에서 예측), 배치·1행 모두 측정 전 1회 미리 실행, 중앙값. 1행 시간에는 호출마다 드는 고정 비용(pandas 입력 검사, 나무 200개 순회 준비)이 포함됨.

  | 모델 | macro-F1 | pickle 크기 | 1000행 배치(행당, 5회) | 1행 예측(50회) |
  |---|---|---|---|---|
  | lightgbm | 0.986 | 918 KB | 0.035 ms | 1.1 ms |
  | random_forest | 0.987 | 2,050 KB | 0.019 ms | 12.9 ms |

  앱은 1분에 1행씩 예측하므로 1행 시간이 기준. 많은 행을 한 번에 예측하면 오히려 랜덤 포레스트가 빠름(호출 1번에 드는 고정 비용이 커서 1행에서 불리함).
  macro-F1 차이 0.001은 사실상 같다고 보고, 크기 약 1/2·1행 예측 약 1/12·텍스트 저장 형식(`focus_lgbm.txt`, pickle 불필요) 때문에 LightGBM을 배포 모델로 둠.
  LightGBM은 비교 전에 배포 후보로 정해 둔 모델이고, 이 표는 그 선택을 측정으로 확인한 것. 측정값은 실행 PC에 따라 달라짐.
  LightGBM은 sklearn 래퍼(`LGBMClassifier`)로 잼. 앱이 `focus_lgbm.txt`를 `lgb.Booster`로 읽어 예측하면 시간이 다를 수 있으니, 앱에 넣을 때 그 경로로 다시 잴 것.
  비교 상대는 실행할 때마다 macro-F1로 다시 고르므로, 재실행 후에는 `deploy_cost`의 키와 이 표가 맞는지 확인.
- 사용자 간 분할과 랜덤 분할 차이가 0.005 → 사용자별 습관보다 규칙 구조를 배운 것.

## 해석·한계
- 점수가 높은 이유는 규칙 라벨이 분 단위로 거의 그대로 이어지기 때문(persistence만으로 0.85).
  모델은 "직전 1분의 규칙 입력 + 최근 15분 맥락"으로 규칙을 복원하는 수준이며, 실제 집중과의 관계는 검증 안 됨.
  self 라벨이 모이기 전에는 앱이 이 모델을 로드하지 않음(규칙 대비 추가 정보가 없음).
- 계획서대로 최종 평가는 DevDesk 자체 self 라벨(~300개)로만 하고, 그때는 날짜 기준 시간 분할로 바꿀 것.
- class_weight="balanced"는 모든 모델에서 손해 → 배포 모델은 가중치 없음.
