# ☁️ Azure MLOps Pipeline & Live Equity Forecaster

[![Live Deployment](https://img.shields.io/badge/Live_App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://stock-return-predictor-usa-mhalam.streamlit.app/)

## 📌 Architecture Overview
This repository contains an end-to-end, cloud-native Machine Learning pipeline designed to forecast daily equity returns. The system integrates real-time market volatility data with live NLP sentiment analysis, utilizing **Microsoft Azure Machine Learning** for strict MLOps tracking and model registry, and a serverless frontend for live inference.

### 🛠️ Core Tech Stack
* **Cloud & MLOps:** Azure ML Workspace, MLflow, Model Registry, Azure Inference Endpoints
* **Machine Learning:** Scikit-learn (Random Forest Classification/Regression), Pandas, NumPy
* **Data Ingestion:** `yfinance` API (Market Data), Finnhub REST API (Live News)
* **NLP Engine:** VADER Sentiment Intensity Analyzer
* **Frontend/Deployment:** Streamlit Cloud

---

## ⚙️ Phase 1: Azure MLOps & Model Lifecycle
The core algorithmic engine was developed, tracked, and versioned entirely within the Azure cloud ecosystem to ensure production-grade reproducibility. 

* **Experiment Tracking:** Utilized `MLflow` context managers to dynamically log training parameters, model signatures, and evaluation metrics (ROC-AUC, RMSE) across multiple concurrent runs.
* **Model Registry & Environment Control:** The highest-performing model was formally registered in the Azure Model Registry. Custom YAML execution environments were defined to lock in strict dependency versions (`scikit-learn==1.5.1`).

### Azure Model Registry & MLflow Tracking
<img width="1913" height="773" alt="image" src="https://github.com/user-attachments/assets/75c34a30-95fb-4c01-ab4c-7000003fb838" />
<img width="1906" height="782" alt="image" src="https://github.com/user-attachments/assets/a2ef6b31-e534-468f-86b3-a0ee32c657d0" />


---

## 🚀 Phase 2: Cloud Deployment & Managed Endpoints
To serve predictions, the registered model artifact was deployed via Azure managed compute to establish a robust backend infrastructure.

* **Inference Endpoints:** Provisioned scalable real-time inference endpoints designed to process low-latency REST API requests.
* **Cost-Optimized Hybrid Serving:** Transitioned the finalized model artifact to a serverless Streamlit architecture for continuous, zero-cost daily inference, simulating a decoupled frontend-backend microservice structure.

### Managed Inference Endpoint Configuration
<img width="1537" height="664" alt="image" src="https://github.com/user-attachments/assets/8d606c3d-7d30-40a6-9c72-75d919d0e8ae" />
<img width="1534" height="651" alt="image" src="https://github.com/user-attachments/assets/cb5de2a5-9c8c-48c5-9829-5b68e9575809" />
<img width="1527" height="658" alt="image" src="https://github.com/user-attachments/assets/c7644837-c0d6-48a9-a5da-703a3f6bc36b" />
<img width="1529" height="658" alt="image" src="https://github.com/user-attachments/assets/3d80cf8e-89c7-4e7b-98b0-ed11697c261f" />


---

## 📊 Phase 3: Real-Time Data & NLP Integration
The production inference script dynamically builds its own feature space upon execution:
1. **Market Volatility:** Pulls the rolling 5-day historical OHLCV data via the `yfinance` API.
2. **Sentiment Extraction:** Queries the Finnhub API for the current day's financial headlines.
3. **NLP Processing:** Passes headlines through the VADER sentiment engine to generate an aggregate sentiment polarity score [-1 to 1].
4. **Live Forecasting:** Feeds the composite feature vector into the localized Azure model artifact to output the predicted next-day close and percentage delta.

---

## 💻 Local Setup & Reproduction
To run this pipeline locally:

1. Clone the repository:
   ```bash
   git clone [https://github.com/MHAlam-2412/Live-ML-OPS-US_stock-predictor](https://github.com/MHAlam-2412/Live-ML-OPS-US_stock-predictor)

2. pip install -r requirements.txt
3. streamlit run app.py
