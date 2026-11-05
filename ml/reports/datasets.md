# ML1 공개 데이터셋 (2026-10-26)

원본은 `ml/data/raw/`(git 제외). 목적: 자체 self 라벨이 쌓이기 전에 ML1(집중 상태 분류기) 파이프라인·특징 검증.

| 데이터 | 위치 | 규모 | 라이선스 | 역할 |
|---|---|---|---|---|
| BEHACOM | `raw/behacom.zip`(원본 전체), `raw/behacom/behacom_slim.csv` | 12명, 1분 단위 167,058행, 84컬럼 | CC BY 4.0 (DOI 10.17632/cg4br62535.2) | **주력**: 앱 범주·전환·입력량 특징, 규칙 라벨 약지도 |
| SWELL-KW (MindPulse 재배포) | `raw/mindpulse-swell-v3/swell_features.tab` | 25명, 1분 단위 3,139행 | 재배포본 라이선스 미기재 → 인용은 원본(Koldijk et al., 2014, DANS DOI 10.17026/DANS-X55-69ZP) | 스트레스·조건 라벨로 전환·입력 특징 검증 |
| Stress by Keystroke/App/Mouse | `raw/stress-detection-by-keystrokeapp-mouse-changes/Data/` | 2명, 5~8일 원시 이벤트, 자기평가 98개 | CC BY-SA 4.0 | 원시 로그 → `activity_samples` 변환 파이프라인 시험 |

## activity_samples 매핑

| activity_samples | BEHACOM | SWELL | Kaggle 2명 |
|---|---|---|---|
| ts (분) | `timestamp`(ms, UTC) | `timestamp`(`20120918T131600000`) | `activewindows.Time` 1분 버킷 |
| exe_name | `current_app` (Linux 사용자는 `.exe` 없음: `firefox`, `soffice.bin`) | 없음 | `App_Name` |
| app_category | exe → 범주 매핑 (분당 `current_app`·`penultimate_app` + `current_app_foreground_time`으로 근사) | 없음 | exe → 범주 매핑 |
| switch_count | `changes_between_apps` | `SnAppChange` (+ `SnTabfocusChange`) | 분 내 `App_Name` 변경 수 |
| idle_sec | 없음 → `keystroke_counter`=0 이고 `mouse_action_counter_0..6` 합=0이면 유휴 1분으로 근사 | 없음 → `SnKeyStrokes`·`SnMouseAct`로 근사 | `inactivity.tsv` `Duration(s)` |
| is_fullscreen | 없음 | 없음 | 없음 |
| 라벨 | 없음 → 규칙 라벨(ide+활발=focus, game=distracted, 유휴 10분↑=break) | `Condition`(R/N/T/I), `Stress` 등 자기평가 | `usercondition.tsv`(피로·스트레스) |

## BEHACOM 줄이기 판단
- 지운 컬럼: `digraph_*`(키 쌍 횟수·시간, 약 11,800), `keystrokes_key_*`, `press_release_average_*`(키별, 약 210).
  키 타이밍 생체인식 특징 → 계획서상 키 입력 후킹은 범위 밖이고, DevDesk 샘플러가 만들 수 없는 특징이라 ML1~ML4 어디에도 못 씀.
- 위 매핑에 필요한 컬럼은 모두 slim에 남음. 없는 값(`is_fullscreen`, 정확한 `idle_sec`)은 원본에도 없음.
- 전체 컬럼이 필요하면 `raw/behacom.zip` + `ml/scripts/slim_behacom.py`의 `DROP`만 바꿔 재생성.

## 데이터 특이사항 (BEHACOM)
- 같은 사용자의 연속 행 간격: ≤70초 127,020 / 70초~10분 39,309 / 10분↑ 717 → 행이 매분 있지는 않음. 10분↑ 공백은 세션 경계(PC 꺼짐)로 처리, 유휴로 세지 말 것.
- 키·마우스 입력이 0인 행 115,973개(69%) → 영상 시청·읽기와 자리 비움을 구분 못 함. break 규칙은 연속 유휴 10분 기준으로.
- 사용자별 양 차이 큼(User2 179분 ~ User7 55,778분) → 사용자별 가중치나 상한 필요.
- 인코딩: 거의 ASCII지만 앱 이름 일부에 Latin-1 바이트(예: `CajaMar-Mam\xe1.txt.exe`) → slim CSV는 UTF-8 + surrogateescape로 쓰고 같은 방식으로 읽기(`ml/behacom.py`).
- 게임 있음(csgo, League of Legends, Steam, Winamax Poker), IDE 적음(sublime_text, java) → focus 규칙 라벨이 적게 나올 수 있음.
- 2026-10-28: `java`/`javaw`는 IDE에서 뺌(→ other). Minecraft 같은 Java 앱도 같은 exe라 focus로 잘못 라벨됨. 이전 `ml/models/` 결과는 java를 IDE로 센 것이라 다시 학습해야 비교 가능(`uv run --group ml python -m ml.train_focus`).

## 한계
실사용 + 집중 라벨이 함께 있는 공개 데스크톱 데이터는 없음 → 최종 평가는 자체 self 라벨로만.
