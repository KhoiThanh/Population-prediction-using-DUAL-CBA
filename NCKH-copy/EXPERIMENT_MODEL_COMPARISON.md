# Bang so sanh ket qua thuc nghiem giua cac mo hinh

Nguon so lieu da co san trong repo:
- artifacts/eval_report.json
- artifacts/rolling_eval_metrics_by_area.csv

Luu y de so sanh cong bang:
- Cung tap du lieu huan luyen va danh gia
- Cung horizon du bao (10 nam)
- Cung giao thuc rolling forecast
- Cung cach loc outlier (MAPE_mean <= 100)

## 1) Bang so sanh cac mo hinh trong de tai

| Mo hinh | MAPE mean filtered (%) | MedAPE mean filtered (%) | n_areas | n_areas filtered | MAE | RMSE | sMAPE | Ghi chu |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Dual-CBA (CNN-BiLSTM + Attention) | 5.9637 | 5.8116 | 2196 | 2092 | N/A | N/A | N/A | So lieu tu eval_report hien co |
| ARIMA | 4.4117 | 4.5239 | 50 | 50 | 368.7810 | 419.9236 | 4.6076 | So lieu smoke benchmark (subset 50 khu vuc) |
| LSTM | 5.1338 | 4.2750 | 50 | 50 | 361.6304 | 398.0639 | 4.7020 | So lieu smoke benchmark (subset 50 khu vuc) |
| GRU | 3.9473 | 3.7506 | 50 | 49 | 342.0642 | 390.0411 | 3.9027 | So lieu smoke benchmark (subset 50 khu vuc) |
| CNN-LSTM | Chua co | Chua co | Chua co | Chua co | Chua co | Chua co | Chua co | Can chay benchmark tren cung protocol |
| Transformer/Temporal model | Chua co | Chua co | Chua co | Chua co | Chua co | Chua co | Chua co | Can chay benchmark tren cung protocol |

Ghi chu: ket qua ARIMA/LSTM/GRU hien tai la ket qua smoke test de xac nhan pipeline, chua phai ket qua full benchmark chinh thuc.

## 2) Bang so sanh voi cong bo khac

| Nguon cong bo | Du lieu | Horizon | Chi so bao cao | Gia tri cong bo | Gia tri de tai (Dual-CBA) | Nhan xet |
|---|---|---|---|---:|---:|---|
| Bai bao A | Dien thong tin | Dien thong tin | MAPE | Dien thong tin | 5.9637 | So sanh khi cung dieu kien |
| Bai bao B | Dien thong tin | Dien thong tin | MAE | Dien thong tin | N/A | Can bo sung MAE trong de tai |
| Bai bao C | Dien thong tin | Dien thong tin | RMSE | Dien thong tin | N/A | Can bo sung RMSE trong de tai |

## 3) Du lieu trich dan nhanh de dien vao bao cao

- MAPE_mean_filtered: 5.963710847955406
- MedAPE_mean_filtered: 5.811635564585541
- n_areas: 2196
- n_areas_filtered: 2092

Bo sung tu smoke benchmark (subset 50 khu vuc):
- ARIMA: MAPE_filtered=4.4117, MedAPE_filtered=4.5239, MAE=368.7810, RMSE=419.9236, sMAPE=4.6076
- LSTM: MAPE_filtered=5.1338, MedAPE_filtered=4.2750, MAE=361.6304, RMSE=398.0639, sMAPE=4.7020
- GRU: MAPE_filtered=3.9473, MedAPE_filtered=3.7506, MAE=342.0642, RMSE=390.0411, sMAPE=3.9027

## 4) Cach dien nhanh bang cho ban

1. Chay tung mo hinh baseline voi cung pipeline tao split va rolling eval.
2. Tinh lai dung bo chi so: MAPE, MedAPE, MAE, RMSE, sMAPE.
3. Dien ket qua vao bang muc 1.
4. Chon 2-4 bai bao gan nhat ve du bao dan so/chuoi thoi gian va dien muc 2.

## 5) Lenh chay benchmark tu dong

Smoke test nhanh (kiem tra pipeline):

```bash
d:/Document/Code/NCKH/.venv/Scripts/python.exe benchmark_models.py --max_areas 50 --epochs 3 --summary_csv artifacts/benchmark_model_comparison_smoke.csv --detail_csv artifacts/benchmark_per_area_metrics_smoke.csv
```

Chay full benchmark de lay ket qua chinh thuc:

```bash
d:/Document/Code/NCKH/.venv/Scripts/python.exe benchmark_models.py --summary_csv artifacts/benchmark_model_comparison.csv --detail_csv artifacts/benchmark_per_area_metrics.csv
```

File ket qua dau ra:
- artifacts/benchmark_model_comparison.csv
- artifacts/benchmark_per_area_metrics.csv
