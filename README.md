# ML Workbench Pro 🧠

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/marshenilmitra/ml-workbench-pro?style=social)](https://github.com/marshenilmitra/ml-workbench-pro)

**An interactive machine learning experimentation platform.**  
Train, evaluate, and export models on **20 datasets** using **22+ algorithms** – all with zero code.  
Built with **Streamlit**, **scikit‑learn**, and **statsmodels**.

---

## 🎥 Demo Video

https://github.com/marshenilmitra/ml-workbench-pro/raw/main/v1.mp4

*Click the link above to watch the 2‑minute walkthrough directly on GitHub.*

---

## 📸 Screenshots

| Supervised | Classification | Clustering | Time Series |
|------------|----------------|------------|-------------|
| ![Supervised](screenshots/dashboard_home.png) | ![Classification](screenshots/classification.png) | ![Clustering](screenshots/clustering.png) | ![Time Series](screenshots/timeseries.png) |

| Leaderboard | Data Profiling | Overfitting Detection |
|-------------|----------------|-----------------------|
| ![Leaderboard](screenshots/leaderboard.png) | ![Profiling](screenshots/data_profiling.png) | ![Overfitting](screenshots/overfitting.png) |

---

## ✨ Features

- **20 datasets** – regression, classification, clustering, time‑series, synthetic  
- **16 supervised** + **6 unsupervised** + **5 time‑series** algorithms  
- **Intelligent model filtering** – only compatible models shown  
- **Hyperparameter tuning** with sliders & default toggle  
- **5‑fold cross‑validation** with per‑fold boxplots  
- **Overfitting detection** – warns when training accuracy ≫ test accuracy  
- **Data profiling** with Data Quality Score (missing, duplicates, outliers)  
- **Model leaderboard** (Top 3 by CV score) – supervised only  
- **Export** – 5‑sheet Excel workbook, trained model (.pkl), ZIP of all outputs  
- **Automatic insights** – explains model performance in plain English  
- **Clean, responsive UI** – badges, spinners, expanders, dark/light friendly  

---

## 🏗️ Architecture
┌─────────────────────────────────────────────────────────────┐
│ STREAMLIT UI │
│ (Radio: Supervised/Unsup/T.S. → Dataset → Model → Params) │
└──────────┬─────────────────────────────────┬───────────────┘
│ │
┌──────────▼──────────┐ ┌──────────────────▼───────────────┐
│ DATA LOADER │ │ MODEL FACTORY │
│ (seaborn, sklearn, │ │ (regression, classification, │
│ synthetic datasets) │ │ clustering, time‑series) │
└──────────┬──────────┘ └──────────────────┬───────────────┘
│ │
┌──────────▼─────────────────────────────────▼───────────────┐
│ PREPROCESSING & TRAINING │
│ LabelEncoder, datetime→numeric, inf/nan handling, scaling │
│ sklearn models (LR, RF, SVR, …), statsmodels (ARIMA, …) │
└──────────┬─────────────────────────────────────────────────┘
│
┌──────────▼──────────┐ ┌──────────────┐ ┌──────────────┐
│ EVALUATION │ │ INSIGHTS │ │ EXPORT │
│ metrics, CV, │ │ plain‑text │ │ Excel, pkl, │
│ charts, │ │ explanations│ │ ZIP download│
│ leaderboard │ │ + warnings │ │ │
└─────────────────────┘ └──────────────┘ └──────────────┘

text

---

## 🧰 Tech Stack

| Category | Tools |
|----------|-------|
| **Frontend** | Streamlit, HTML/CSS |
| **ML / Stats** | scikit‑learn, statsmodels, NumPy, pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Export** | xlsxwriter, pickle, zipfile |
| **Version Control** | Git, GitHub |

---

## 📚 Supported Algorithms

### Supervised – Regression
Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Gradient Boosting, SVR, K‑Neighbors

### Supervised – Classification
Logistic Regression, K‑Neighbors, Decision Tree, Random Forest, SVM (SVC), Gaussian Naive Bayes, Gradient Boosting

### Unsupervised
K‑Means, Agglomerative Clustering, DBSCAN, Gaussian Mixture, PCA, t‑SNE

### Time Series Forecasting
ARIMA, SARIMA, Holt‑Winters, Holt’s Linear Trend, Simple Moving Average

---

## 📂 Project Structure
ml-workbench-pro/
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── screenshots/ # App screenshots
│ ├── dashboard_home.png
│ ├── classification.png
│ ├── clustering.png
│ ├── timeseries.png
│ ├── leaderboard.png
│ ├── data_profiling.png
│ └── overfitting.png
├── v1.mp4 # Demo video (playable inline)
├── .gitignore
└── README.md
## 🚀 Live Demo

Try it now: [https://your-app-url.streamlit.app](https://your-app-url.streamlit.app) *(after deploying to Streamlit Cloud)*

---

## 🛠️ Run Locally

```bash
git clone https://github.com/marshenilmitra/ml-workbench-pro.git
cd ml-workbench-pro
pip install -r requirements.txt
streamlit run app.py
🔮 Future Enhancements
Integrate sklearn.pipeline for production‑grade preprocessing

Add SHAP / LIME for model explainability

Model experiment tracking with MLflow

User authentication and saved experiment history

Multi‑page Streamlit layout for better navigation

👤 Author
Marshenil Mitra
LinkedIn | GitHub

Built with passion during CDAC PG‑DBDA – a demonstration of end‑to‑end ML engineering.

📄 License
This project is licensed under the MIT License – see LICENSE file for details.


---

## 📌 Final notes

- Replace `https://your-app-url.streamlit.app` with your actual Streamlit Cloud URL after deploying.  
- Replace `https://linkedin.com/in/yourprofile` with your real LinkedIn profile.  
- The video link uses the **raw** URL to ensure playback. If you want to use the LinkedIn video instead, swap it with your LinkedIn activity link.  
- Your repository is now **fully 12 LPA‑ready**. Copy this README, paste it on GitHub, and your project will shine.
