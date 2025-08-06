# 배터리 텔레메트리 예측

이 프로젝트는 딥러닝 모델을 사용하여 배터리의 상태(SOC, State of Charge), 가용 전력, 셀 온도(최고, 평균, 최저) 등 주요 텔레메트리 데이터를 예측하는 것을 목표로 합니다. 원본 데이터 전처리부터 모델 학습, 평가, 추론에 이르는 전체 파이프라인을 제공합니다.

## 주요 특징

- **다중 작업 예측(Multi-Task Prediction)**: SOC(%), 가용 셀 전력(kW), 미래의 온도 시퀀스를 동시에 예측합니다.
- **다양한 모델 아키텍처 지원**: TCN(Temporal Convolutional Network), LSTM, Transformer 백본을 지원합니다.
- **End-to-End 파이프라인**:
    - **전처리**: 원본 BMS 데이터를 정제하고, 이상치를 처리하며, 파생 변수를 생성하고, 데이터를 스케일링합니다.
    - **윈도잉**: 처리된 시계열 데이터로부터 모델 학습을 위한 시퀀스 윈도우를 생성합니다.
    - **학습**: 유연한 설정을 통해 선택된 모델을 학습시킵니다.
    - **평가**: 상세한 성능 지표(MAE, RMSE, R² 등), 예측 구간별 분석, 시각화 자료를 제공합니다.
- **설정 기반 제어**: 데이터 경로, 모델 하이퍼파라미터 등 파이프라인의 모든 측면을 중앙 `config.yaml` 파일 하나로 제어합니다.

## 프로젝트 구조

```
.
├── artifacts/            # 모델 체크포인트, 스케일러, 평가 결과, 플롯 저장
├── data/
│   ├── raw/              # 원본 BMS .csv 파일 위치
│   ├── interim/          # 정제 및 병합된 데이터 저장
│   └── processed/        # 최종 윈도우 데이터셋(.npz) 저장
├── src/                  # 소스 코드
│   ├── models/           # 모델 정의 (TCN, LSTM, Transformer)
│   ├── preprocess.py     # 원본 데이터 정제 및 피처 엔지니어링
│   ├── labels.py         # 윈도우 및 라벨 생성
│   ├── train.py          # 모델 학습 스크립트
│   ├── evaluate.py       # 종합 모델 평가 스크립트
│   ├── infer.py          # 간단한 추론 및 예측 스크립트
│   └── config.py         # 설정 로더
├── config.yaml           # 메인 설정 파일
├── requirements.txt      # Python 의존성 파일
└── README.md             # 본 파일
```

## 설치 방법

1.  저장소를 복제(Clone)합니다.
2.  필요한 Python 패키지를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```

## 사용 방법

전체 작업 흐름은 `config.yaml` 파일을 통해 제어됩니다. 스크립트를 실행하기 전, `config.yaml` 파일을 열어 데이터 경로, 세션 분할, 모델 파라미터 등을 자신의 환경에 맞게 수정하세요.

**1단계: `config.yaml` 설정**

-   `raw_dir`, `processed_dir` 등 데이터 경로를 설정합니다.
-   학습, 검증, 테스트에 사용할 세션 이름을 정의합니다 (`sessions_train`, `sessions_val`, `sessions_test`).
-   윈도우 파라미터(`past_seconds`, `horizon_seconds`, `dt`)를 조정합니다.
-   사용할 모델 아키텍처(`tcn`, `lstm`, `transformer`)와 학습 하이퍼파라미터를 선택하고 설정합니다.

**2단계: 원본 데이터 전처리**

이 스크립트는 `data/raw/`에 있는 원본 CSV 파일을 읽어 정제하고, 피처를 생성한 뒤 `data/interim/merged_clean.csv` 파일로 저장합니다. 또한 학습 데이터셋 기준으로 스케일러를 학습시키고 파일로 저장합니다.

```bash
python -m src.preprocess --config config.yaml
```

**3단계: 윈도우 및 라벨 생성**

전처리된 데이터를 입력받아 입력 윈도우(`X`)와 예측 대상(`Y`, 라벨)을 생성하고, `data/processed/` 경로에 압축된 NumPy 파일(`.npz`)로 저장합니다.

```bash
python -m src.labels --config config.yaml
```

**4단계: 모델 학습**

설정 파일에 지정된 모델을 학습시킵니다. 커맨드 라인 인자를 통해 모델을 변경할 수 있습니다.

```bash
# TCN 모델 학습 (config.yaml에 지정되었거나 기본값)
python -m src.train --model_name tcn --config config.yaml

# 또는 LSTM 모델 학습
python -m src.train --model_name lstm --config config.yaml
```
체크포인트와 로그는 `artifacts/<model_name>/` 디렉토리에 저장됩니다.

**5단계: 모델 평가**

학습된 모델을 `val` 또는 `test` 세트에서 상세히 평가합니다. 이 과정에서 주요 지표, 분석 결과, 플롯이 포함된 `qcd_report.json` 파일이 생성됩니다.

```bash
python -m src.evaluate --model_name tcn --split test --config config.yaml
```

**6단계: 추론 실행**

특정 데이터 스플릿에 대한 예측 결과를 얻으려면 `infer.py` 스크립트를 사용합니다.

```bash
python -m src.infer --model_name tcn --split test --config config.yaml
```