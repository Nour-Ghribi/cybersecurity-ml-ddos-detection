# Detecting DDoS Attacks using LSTM on the CIC-DDoS2019 dataset

This project provides a clean, reproducible, end-to-end pipeline for:

- Data exploration
- Data cleaning & preprocessing
- Model training
- Model testing
- Model evaluation
- Tuning & optimization

It follows an incremental development strategy:

1. `phase1_dns_binary`
2. `phase2_multiclass_subset`
3. `phase3_scaling_subset`

## Run

```powershell
python ddos_lstm_project.py --base-dir . --output-dir outputs
```

Run a single phase:

```powershell
python ddos_lstm_project.py --base-dir . --output-dir outputs --phase phase1_dns_binary
```
