# Prediksi Biaya Pengobatan Pasien Menggunakan XGBoost dengan Pendekatan Explainable AI

**Student:** Ammar Pavel Zamora Siregar (1202224044)  
**Program:** Sarjana Informatika, Universitas Telkom  
**Year:** 2025

## Project Overview
This thesis project implements XGBoost with Explainable AI (SHAP & LIME) for patient treatment cost prediction using the Kaggle Insurance Cost dataset.

## Current Status: Phase 0 - Environment Setup ✅

## Dataset
- **Source:** Kaggle Insurance Cost Dataset
- **Records:** 1,338 patients
- **Features:** 7 (age, sex, bmi, children, smoker, region, charges)
- **Target:** Medical charges (treatment costs)

## Project Phases
- [x] **Phase 0:** Environment Setup & GitHub Repository
- [ ] **Phase 1:** Data Analysis & Baseline (Week 1)
- [ ] **Phase 2:** XGBoost Implementation (Week 2)
- [ ] **Phase 3:** Explainable AI Integration (Week 3)
- [ ] **Phase 4:** Dashboard Development (Week 4)
- [ ] **Phase 5:** Documentation & Paper Completion (Week 5-6)

## Setup Instructions

1. **Clone Repository:**
   ```bash
   git clone https://github.com/yourusername/thesis-xgboost-explainable-ai.git
   cd thesis-xgboost-explainable-ai
   ```
2. **Create & activate venv**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```  
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Add dataset**
   ```bash
   - Download the Kaggle Insurance Cost dataset from [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance).
   - Place the dataset in the `data/` directory.
   ```
5. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```