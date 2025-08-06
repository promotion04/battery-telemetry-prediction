# play.bash
#!/usr/bin/env bash
set -euo pipefail

# 0) 1회 공통: 전처리/라벨
python -m src.preprocess --config config.yaml
python -m src.labels     --config config.yaml

# 1) Train (3 models)
python -m src.train --config config.yaml --model_name lstm
python -m src.train --config config.yaml --model_name transformer
python -m src.train --config config.yaml --model_name tcn

# 2) LSTM (val, test)
python -m src.evaluate --config config.yaml --model_name lstm --split val
python -m src.infer    --config config.yaml --model_name lstm --split val --bench 10
python -m src.evaluate --config config.yaml --model_name lstm --split test
python -m src.infer    --config config.yaml --model_name lstm --split test --bench 10
python -m src.bench_test_six_segments --config config.yaml --model_name lstm

# 3) Transformer (val, test)
python -m src.evaluate --config config.yaml --model_name transformer --split val
python -m src.infer    --config config.yaml --model_name transformer --split val --bench 10
python -m src.evaluate --config config.yaml --model_name transformer --split test
python -m src.infer    --config config.yaml --model_name transformer --split test --bench 10
python -m src.bench_test_six_segments --config config.yaml --model_name transformer

# 4) TCN (val, test)
python -m src.evaluate --config config.yaml --model_name tcn --split val
python -m src.infer    --config config.yaml --model_name tcn --split val --bench 10
python -m src.evaluate --config config.yaml --model_name tcn --split test
python -m src.infer    --config config.yaml --model_name tcn --split test --bench 10
python -m src.bench_test_six_segments --config config.yaml --model_name tcn

# 5) 하이퍼파라미터 스윕 (시퀀스 메트릭 기준)
# TCN
python -m src.sweep_hparam_mae --config config.yaml --model_name tcn --key model.tcn.kernel_size --values 1,3,5,7,9 --metric th_seq_MAE --split val --epochs 10
python -m src.sweep_hparam_mae --config config.yaml --model_name tcn --key model.tcn.dropout      --values 0.00,0.05,0.10,0.20,0.30 --metric th_seq_MAE --split val --epochs 10
python -m src.sweep_hparam_mae --config config.yaml --model_name tcn --key model.tcn.levels       --values 2,3,4,5 --metric th_seq_MAE --split val --epochs 10
python -m src.sweep_hparam_mae --config config.yaml --model_name tcn --key model.tcn.hidden       --values 32,64,96,128 --metric pcell_seq_MAE --split val --epochs 10
# JSON 배열은 bash에선 작은따옴표
python -m src.sweep_hparam_mae --config config.yaml --model_name tcn --key model.tcn.num_channels --values '[[64,64],[64,64,64],[64,96,96],[96,96,96]]' --metric th_seq_MAE --split val --epochs 10

# LSTM
python -m src.sweep_hparam_mae --config config.yaml --model_name lstm --key model.lstm.hidden_size --values 32,64,128,256 --metric soc_seq_MAE --split val --epochs 10
python -m src.sweep_hparam_mae --config config.yaml --model_name lstm --key model.lstm.num_layers  --values 1,2,3 --metric th_seq_MAE --split val --epochs 10
python -m src.sweep_hparam_mae --config config.yaml --model_name lstm --key model.lstm.dropout     --values 0.00,0.10,0.20,0.30 --metric pcell_seq_MAE --split val --epochs 10

# Transformer  (주의: d_model % nhead == 0)
python -m src.sweep_hparam_mae --config config.yaml --model_name transformer --key model.transformer.d_model            --values 64,128,256 --metric th_seq_MAE --split val --epochs 10
python -m src.sweep_hparam_mae --config config.yaml --model_name transformer --key model.transformer.nhead              --values 2,4,8 --metric soc_seq_MAE --split val --epochs 10
python -m src.sweep_hparam_mae --config config.yaml --model_name transformer --key model.transformer.num_encoder_layers --values 2,3,4 --metric pcell_seq_MAE --split val --epochs 10


