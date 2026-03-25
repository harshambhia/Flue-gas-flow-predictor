# 🌡️ Flue Gas Flow Rate Predictor — MLP Neural Network

A Streamlit web app that trains an MLP (Multi-Layer Perceptron) regressor to predict **total flue gas flow rate** from 5 input parameters.

## Features
- Upload CSV or Excel datasets (first 5 columns = inputs, 6th = output)
- 70/30 train-test split with StandardScaler normalisation
- Configurable hidden layers, neurons, activation, learning rate, regularisation
- **Parity Plot** — actual vs predicted scatter (train + test)
- **Validation Plot** — loss curve + residuals
- **Series Comparison** — actual vs predicted time-series for both splits
- Single-point prediction panel
- Download predictions as CSV

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → select `app.py` → Deploy

## Dataset format

| Col 1 | Col 2 | Col 3 | Col 4 | Col 5 | Col 6 (target) |
|-------|-------|-------|-------|-------|----------------|
| Input 1 | Input 2 | Input 3 | Input 4 | Input 5 | Total Flow Rate |

---
Built with Streamlit · scikit-learn · Matplotlib
