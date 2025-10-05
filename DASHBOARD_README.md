# 🏥 Healthcare Cost Predictor - Streamlit Dashboard

**Bachelor Thesis Project**
Student: Ammar Pavel Zamora Siregar (1202224044)
Universitas Telkom, 2025

---

## 📋 Dashboard Overview

Interactive web application for healthcare cost prediction using XGBoost ensemble with dual Explainable AI framework (SHAP + LIME).

### Features:
- 🔮 **Cost Prediction**: Real-time healthcare cost estimates with confidence intervals
- 📊 **LIME Explanations**: Patient-specific local explanations (~8 seconds)
- 🔍 **SHAP Analysis**: Global feature impact analysis with waterfall plots
- 💡 **What-If Scenarios**: Smoking cessation & weight management impact calculator
- 🎯 **Risk Assessment**: Automatic risk categorization (Low/Medium/High)
- 📈 **Comparison Metrics**: vs Population/Smoker/Non-Smoker averages

---

## 🚀 Local Testing

### Prerequisites
- Python 3.11+
- Virtual environment activated

### Run Locally:

```bash
# 1. Ensure virtual environment is activated
venv\Scripts\activate  # Windows

# 2. Install dependencies (if not already)
pip install -r requirements.txt

# 3. Run Streamlit app
streamlit run app.py

# 4. Open browser at http://localhost:8501
```

### Expected Output:
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

---

## ☁️ Streamlit Cloud Deployment

### Step 1: Prepare GitHub Repository

**Ensure these files exist:**
```
thesis-xgboost-explainable-ai/
├── app.py                              ✅ Main Streamlit app
├── requirements.txt                    ✅ Dependencies
├── .streamlit/
│   └── config.toml                     ✅ Streamlit config
├── results/
│   └── models/
│       └── final_best_model.pkl        ✅ Trained model (CRITICAL)
└── data/
    └── processed/
        └── insurance_enhanced_processed.csv  ✅ Training data (for SHAP/LIME)
```

**IMPORTANT: Check file sizes**
- GitHub has 100MB file size limit
- If `final_best_model.pkl` > 100MB, use Git LFS:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "results/models/*.pkl"
git add .gitattributes

# Commit and push
git add .
git commit -m "Add Streamlit dashboard with model artifacts"
git push origin main
```

### Step 2: Deploy to Streamlit Cloud

1. **Go to**: https://share.streamlit.io/

2. **Sign in** with your GitHub account

3. **Click "New app"**

4. **Fill in deployment form:**
   - Repository: `ammarpvl29/thesis-xgboost-explainable-ai`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL (custom): `healthcare-cost-predictor` (or your choice)

5. **Advanced settings** (optional):
   - Python version: `3.11`
   - Click "Deploy!"

6. **Wait for deployment** (~2-5 minutes)

### Step 3: Verify Deployment

**Your dashboard will be live at:**
```
https://ammarpvl29-thesis-xgboost-ex-healthcare-cost-predictor.streamlit.app
```
(Exact URL depends on your choices)

**Test the app:**
- ✅ Patient input form works
- ✅ Prediction generates successfully
- ✅ LIME explanation displays (~8 seconds)
- ✅ SHAP explanation displays
- ✅ What-If scenarios calculate correctly

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'shap'"
**Solution:** Ensure `requirements.txt` is properly formatted and committed

### Issue: "FileNotFoundError: final_best_model.pkl"
**Solution:**
1. Check model file exists: `results/models/final_best_model.pkl`
2. Verify file is pushed to GitHub (check repository online)
3. If >100MB, use Git LFS (see above)

### Issue: "MemoryError" during SHAP computation
**Solution:** This is a known issue with Streamlit Cloud's free tier (1GB RAM limit)
- Reduce SHAP background samples from 100 to 50 in `app.py` line ~407:
  ```python
  background = shap.sample(X_train, 50, random_state=42)  # Changed from 100
  ```

### Issue: LIME takes >30 seconds
**Solution:** Reduce num_samples from 5000 to 2000 in `app.py` line ~371:
```python
num_samples=2000  # Changed from 5000
```

### Issue: App crashes on startup
**Solution:** Check Streamlit Cloud logs:
1. Go to your app dashboard
2. Click "Manage app" → "Logs"
3. Look for error messages
4. Common fixes:
   - Missing dependencies → add to `requirements.txt`
   - File path errors → use relative paths
   - Model compatibility → re-save model with same scikit-learn version

---

## 📊 Dashboard Usage Guide

### For Patients:

1. **Enter Your Information:**
   - Age (18-64)
   - Sex (Male/Female)
   - BMI (15-55)
   - Number of children (0-5)
   - Smoking status (Yes/No)
   - Region (Northeast/Northwest/Southeast/Southwest)

2. **Click "Predict Healthcare Cost"**

3. **Review Your Results:**
   - **Estimated Cost**: Your predicted annual healthcare cost
   - **Risk Category**: Low (🟢) / Medium (🟠) / High (🔴)
   - **Comparison**: How you compare to population/smoker averages

4. **Understand What's Driving Your Cost:**
   - **LIME Tab**: See YOUR specific cost breakdown
   - **SHAP Tab**: See how your features compare globally
   - **What-If Tab**: Calculate savings from lifestyle changes

5. **Take Action:**
   - If smoker: See exact savings from quitting (~$8,000-$23,600)
   - If overweight/obese: See savings from weight loss
   - Combined: See maximum potential savings (~$45,200)

### For Researchers/Conference Attendees:

**Key Technical Highlights to Demonstrate:**

1. **Model Performance:**
   - R² = 0.8770 (exceeds thesis target ≥0.87)
   - Low overfitting (gap = 0.0102)
   - Fast predictions (milliseconds)

2. **Dual XAI Framework:**
   - **SHAP**: Global consistency, mathematically grounded
   - **LIME**: Local explanations, patient-friendly
   - **Complementarity**: SHAP for validation, LIME for interaction

3. **Patient Empowerment:**
   - Quantified savings (not just "reduce costs")
   - Actionable recommendations (smoking, weight)
   - Clear visualizations (no ML jargon)

4. **Production-Ready:**
   - Real-time explanations (~8s LIME)
   - Confidence intervals
   - Risk stratification

---

## 📝 Presentation Tips (Conference Demo)

### 5-Minute Demo Flow:

**Minute 1: Problem Statement**
- "92% of patients want cost transparency, but it's rarely available"
- "We built an AI system that's both accurate AND explainable"

**Minute 2: Live Prediction**
- Enter a HIGH-RISK patient (e.g., Age=50, BMI=35, Smoker=Yes)
- Show prediction: ~$40,000-$50,000
- Highlight risk category: 🔴 High Risk

**Minute 3: Explainability (LIME)**
- Show LIME tab: "Here's exactly what drives THIS patient's cost"
- Point to smoking-BMI interaction: +$15,000-$18,000
- "This is patient-friendly, real-time (~8 seconds)"

**Minute 4: What-If Scenarios**
- Toggle smoking cessation: "If you quit smoking..."
- Show dramatic reduction: $40,000 → $12,000 (~$28,000 savings!)
- "This motivates behavior change with concrete numbers"

**Minute 5: Key Achievements**
- R² = 0.8770 (competitive with state-of-the-art)
- Dual XAI: SHAP (global) + LIME (local)
- Patient empowerment: Quantified savings, actionable insights
- "This is deployed and accessible to anyone, anywhere"

### Questions to Anticipate:

**Q: Why both SHAP and LIME?**
A: They're complementary. SHAP provides global consistency and model validation (for researchers/regulators). LIME provides fast, intuitive local explanations (for patients). We empirically demonstrated they serve different but synergistic roles.

**Q: How accurate are the savings estimates?**
A: Our model achieves R² = 0.8770 with 95% confidence intervals. Savings are calculated by re-running prediction with modified inputs. They're estimates based on 1,338 patient historical data, not guarantees.

**Q: Can this work in Indonesia/other countries?**
A: The methodology is transferable. We'd need to retrain with local healthcare data (costs, demographics, regional factors). The dual XAI framework and patient-centric design principles remain applicable.

**Q: What about data privacy?**
A: The dashboard runs entirely client-side for demo purposes. For production, we'd implement: (1) No data storage, (2) HTTPS encryption, (3) Compliance with local regulations (GDPR, HIPAA, etc.). Current demo doesn't collect or store any patient information.

---

## 🎓 Academic Context

**Thesis Title:** Prediksi Biaya Pengobatan Pasien Menggunakan XGBoost dengan Pendekatan Explainable AI

**Research Contributions:**

1. **Methodological:**
   - Domain-informed preprocessing (WHO BMI standards)
   - Systematic optimization (baseline → optimization → ensemble)
   - Dual XAI framework demonstrating SHAP-LIME complementarity

2. **Empirical:**
   - Smoking dominance quantified (280% cost increase)
   - BMI-smoking synergy validated (370% for obese smokers)
   - 100% correlation: top 5% high-cost cases are smokers (67/67)

3. **Practical:**
   - Patient empowerment with quantified savings
   - Production-ready deployment (Streamlit Cloud)
   - Real-time explainability (~8s LIME, ~3-5s SHAP)

**Dataset:** Kaggle Insurance Cost (1,338 patients)
**Performance:** R² = 0.8770, RMSE = $4,320
**Technology:** XGBoost + SHAP + LIME + Streamlit

---

## 📧 Contact & Support

**Student:** Ammar Pavel Zamora Siregar
**NIM:** 1202224044
**Email:** ammarpvl@student.telkomuniversity.ac.id
**Institution:** Universitas Telkom, School of Informatics

**Supervisors:**
- Indra Aulia, S.TI., M.Kom.
- Nurul Ilmi, S.Kom, M.T

**GitHub Repository:** https://github.com/ammarpvl29/thesis-xgboost-explainable-ai

---

## 🔗 Useful Links

- **Streamlit Documentation:** https://docs.streamlit.io/
- **Streamlit Cloud:** https://share.streamlit.io/
- **SHAP Documentation:** https://shap.readthedocs.io/
- **LIME Documentation:** https://lime-ml.readthedocs.io/
- **XGBoost Documentation:** https://xgboost.readthedocs.io/

---

**Last Updated:** October 5, 2025
**Dashboard Version:** 1.0.0
**Status:** Ready for Conference Presentation 🚀
