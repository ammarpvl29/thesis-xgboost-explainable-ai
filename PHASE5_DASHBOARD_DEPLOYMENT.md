# 📱 Phase 5: Dashboard Deployment & Conference Paper - Complete Documentation

**Date:** October 5, 2025
**Status:** ✅ COMPLETE
**Student:** Ammar Pavel Zamora Siregar (1202224044)
**Supervisors:** Achmad Udin Zailani, S.Kom., M.Kom. & Nurul Ilmi, S.Kom, M.T

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Streamlit Dashboard Implementation](#streamlit-dashboard-implementation)
3. [Critical Bug Fixes](#critical-bug-fixes)
4. [Conference Paper Creation](#conference-paper-creation)
5. [Deployment Process](#deployment-process)
6. [File Structure](#file-structure)
7. [Technical Implementation Details](#technical-implementation-details)
8. [How to Resume Work](#how-to-resume-work)

---

## 🎯 Overview

### What Was Accomplished Today:

✅ **Dashboard Development:**
- Built complete Streamlit web application (`app.py`)
- Integrated final ensemble model (R² = 0.8770)
- Implemented SHAP global explanations
- Implemented LIME local explanations
- Created what-if scenario analysis (smoking cessation, weight management)

✅ **Critical Bug Fixes:**
- Fixed categorical encoding mismatch between training and dashboard
- Resolved SHAP/LIME compatibility issues

✅ **Conference Paper:**
- Created IEEE conference paper (`conference-paper.tex`)
- Updated supervisor information (Achmad Udin Zailani)

✅ **Deployment:**
- Deployed to Streamlit Cloud
- Live URL: https://healthcare-cost-predictor.streamlit.app/

---

## 🖥️ Streamlit Dashboard Implementation

### Dashboard File: `app.py`

**Key Features Implemented:**

#### 1. **Patient Input Form (Sidebar)**
```python
# Inputs:
- Age: Slider (18-64)
- Sex: Radio button (Male/Female)
- BMI: Number input (15-55)
- Children: Selectbox (0-5)
- Smoker: Radio button (Yes/No)
- Region: Dropdown (Northeast/Northwest/Southeast/Southwest)
```

#### 2. **Real-Time Cost Prediction**
- **Model Loading:** Cached ensemble model from `results/models/final_best_model.pkl`
- **Confidence Intervals:** 95% CI using ensemble base estimators std
- **Risk Categorization:** Low (🟢) / Medium (🟠) / High (🔴)
- **Comparison Metrics:**
  - vs Population Average ($13,270)
  - vs Smoker/Non-Smoker averages

#### 3. **SHAP Explanations Tab**
- **Explainer Type:** `shap.Explainer` with PermutationExplainer
- **Background Sample:** 100 data points from training set
- **Visualization:** Waterfall plot showing feature contributions
- **Feature Impact Table:** Top 10 features with SHAP values

**Implementation:**
```python
# Prepare encoded training data
X_train_encoded = prepare_encoded_training_data(training_data)
background = shap.sample(X_train_encoded, 100, random_state=42)

# Initialize SHAP explainer
shap_explainer = shap.Explainer(model.predict, background)
shap_values = shap_explainer(patient_features)

# Display waterfall plot
shap.waterfall_plot(shap_values[0], show=False)
```

#### 4. **LIME Explanations Tab**
- **Explainer Type:** `LimeTabularExplainer`
- **Num Samples:** 5,000 perturbations per patient
- **Computation Time:** ~8 seconds per patient
- **Visualization:** Feature contribution bar plot
- **Output:** Top cost drivers/reducers with $ impact

**Implementation:**
```python
# Initialize LIME explainer
lime_explainer = LimeTabularExplainer(
    training_data=X_train_encoded.values,
    feature_names=FEATURE_COLS,
    mode='regression',
    discretize_continuous=True,
    random_state=42
)

# Generate explanation
lime_exp = lime_explainer.explain_instance(
    data_row=patient_features.values[0],
    predict_fn=model.predict,
    num_features=10,
    num_samples=5000
)
```

#### 5. **What-If Scenario Analysis Tab**
- **Smoking Cessation Impact:**
  - Toggle "What if I quit smoking?"
  - Shows new predicted cost
  - Displays savings amount ($)

- **Weight Management Impact:**
  - Target BMI slider
  - Predicted cost at target BMI
  - Potential savings calculation

- **Combined Intervention:**
  - Total savings from smoking + weight loss
  - Maximum lifestyle impact demonstration

**Implementation:**
```python
# Smoking cessation scenario
modified_features = engineer_features(age, sex, bmi, children, 'No', region)
modified_pred, _, _ = predict_cost(model, modified_features)
savings = prediction - modified_pred

# Weight management scenario
weight_features = engineer_features(age, sex, target_bmi, children, smoker, region)
weight_pred, _, _ = predict_cost(model, weight_features)
weight_savings = prediction - weight_pred
```

---

## 🐛 Critical Bug Fixes

### Issue 1: Categorical Encoding Mismatch

**Error Encountered:**
```
Error generating LIME explanation: could not convert string to float: 'female'
Error generating SHAP explanation: could not convert string to float: 'female'
```

**Root Cause:**
- Training data used **lowercase** categorical values ('male', 'female', 'yes', 'no', 'northeast', etc.)
- Training encoding used `pd.Categorical(X[col]).codes` which creates **alphabetical encoding**
- Dashboard received **capitalized** inputs ('Male', 'Female', 'Yes', 'No')
- Custom encoding maps didn't match training encoding

**Solution Implemented:**

1. **Updated `engineer_features()` function:**
```python
def engineer_features(age, sex, bmi, children, smoker, region):
    # Convert inputs to lowercase to match training data
    sex_lower = sex.lower()
    smoker_lower = smoker.lower()
    region_lower = region.lower()

    # Categorical encoding using pd.Categorical().codes logic (alphabetical)
    # sex: female=0, male=1
    sex_encoded = 0 if sex_lower == 'female' else 1

    # region: northeast=0, northwest=1, southeast=2, southwest=3
    region_map_lower = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
    region_encoded = region_map_lower.get(region_lower, 0)

    # bmi_category: Normal=0, Obese=1, Overweight=2, Underweight=3 (alphabetical)
    bmi_cat_map = {'Normal': 0, 'Obese': 1, 'Overweight': 2, 'Underweight': 3}

    # age_group: 18-29=0, 30-39=1, 40-49=2, 50-64=3 (alphabetical)
    age_grp_map = {'18-29': 0, '30-39': 1, '40-49': 2, '50-64': 3}
```

2. **Created `prepare_encoded_training_data()` function:**
```python
@st.cache_data
def prepare_encoded_training_data(_training_data):
    """
    Encode training data EXACTLY as it was encoded during model training.
    Uses pd.Categorical().codes which creates alphabetical encoding.
    """
    X = _training_data[FEATURE_COLS].copy()

    # Encode categorical columns using pd.Categorical().codes
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_cols:
        X[col] = pd.Categorical(X[col]).codes

    X = X.astype(float)
    return X
```

3. **Updated SHAP/LIME sections to use properly encoded data:**
```python
# LIME section
X_train_encoded = prepare_encoded_training_data(training_data)
lime_explainer = LimeTabularExplainer(
    training_data=X_train_encoded.values,  # Properly encoded
    ...
)

# SHAP section
X_train_encoded = prepare_encoded_training_data(training_data)
background = shap.sample(X_train_encoded, 100, random_state=42)  # Properly encoded
```

**Key Learning:**
- **ALWAYS match training preprocessing exactly** - including:
  - Value casing (lowercase vs uppercase)
  - Encoding method (pd.Categorical().codes vs custom maps)
  - Feature order
  - Data types (float64)

---

## 📄 Conference Paper Creation

### File: `conference-paper.tex`

**Paper Structure (IEEE Conference Format):**

1. **Title:** "Interpretable Healthcare Cost Prediction Using XGBoost with Dual Explainable AI Framework"

2. **Authors:**
   - Ammar Pavel Zamora Siregar (Student)
   - Achmad Udin Zailani, S.Kom., M.Kom. (Supervisor 1)
   - Nurul Ilmi, S.Kom, M.T (Supervisor 2)

3. **Abstract (Key Numbers):**
   - Dataset: 1,338 patients
   - Performance: R² = 0.8770
   - SHAP: $6,397.52 smoking-BMI impact
   - LIME: $74,518 lifestyle delta
   - Savings: $8,000-$45,200 potential

4. **Main Sections:**
   - Introduction (problem statement, contributions)
   - Related Work (XGBoost, XAI, research gap)
   - Methodology (preprocessing, optimization, ensemble, SHAP/LIME)
   - Results (4 tables with performance metrics)
   - Discussion (validation, comparison, limitations)
   - Conclusion (achievements, future work)

5. **Tables Included:**
   - Table I: Model Performance Evolution
   - Table II: SHAP Global Feature Importance (Top 5)
   - Table III: LIME Patient Profile Analysis
   - (SHAP-LIME complementarity covered in text)

6. **References:** 11 IEEE-formatted citations

**How to Compile:**
```bash
# Method 1: Local LaTeX
pdflatex conference-paper.tex
bibtex conference-paper
pdflatex conference-paper.tex
pdflatex conference-paper.tex

# Method 2: Overleaf (recommended)
# Upload conference-paper.tex to https://www.overleaf.com
# Click "Recompile" → Download PDF
```

---

## 🚀 Deployment Process

### Streamlit Cloud Deployment Steps:

1. **Files Prepared:**
   - ✅ `app.py` - Main Streamlit application
   - ✅ `requirements.txt` - Dependencies (streamlit, shap, lime, etc.)
   - ✅ `.streamlit/config.toml` - Streamlit configuration
   - ✅ `results/models/final_best_model.pkl` - Trained model
   - ✅ `data/processed/insurance_enhanced_processed.csv` - Training data

2. **Deployment Platform:**
   - Platform: Streamlit Community Cloud (FREE)
   - URL: https://healthcare-cost-predictor.streamlit.app/
   - Auto-deploy: Enabled (triggers on git push)

3. **Git Push Workflow:**
```bash
git add app.py requirements.txt .streamlit/ conference-paper.tex README.md
git commit -m "Add Streamlit dashboard and conference paper"
git push origin main
```

4. **Streamlit Cloud Auto-Deploy:**
   - Detects changes on GitHub
   - Rebuilds app automatically
   - Deployment time: 1-3 minutes
   - Status: https://share.streamlit.io/ (check app dashboard)

**Important Notes:**
- ✅ No manual redeploy needed - auto-deploys on git push
- ✅ Model file (<100MB) works fine with standard git
- ✅ If model >100MB, use Git LFS: `git lfs track "*.pkl"`

---

## 📁 File Structure

### New Files Created Today:

```
thesis-xgboost-explainable-ai/
├── app.py                              # ✅ NEW - Streamlit dashboard
├── conference-paper.tex                # ✅ NEW - IEEE conference paper
├── requirements.txt                    # ✅ UPDATED - Added streamlit
├── .streamlit/
│   └── config.toml                     # ✅ NEW - Streamlit config
├── DASHBOARD_README.md                 # ✅ NEW - Dashboard documentation
├── PHASE5_DASHBOARD_DEPLOYMENT.md      # ✅ NEW - This file
└── README.md                           # ✅ UPDATED - Phase 5 complete
```

### Key Files for Dashboard:

1. **`app.py`** (Main application)
   - Patient input form
   - Prediction logic with feature engineering
   - SHAP/LIME integration
   - What-if scenario analysis
   - UI/UX with custom CSS

2. **`requirements.txt`** (Dependencies)
   - streamlit>=1.28.0
   - shap>=0.43.0
   - lime>=0.2.0.1
   - All ML libraries (xgboost, lightgbm, scikit-learn)

3. **`.streamlit/config.toml`** (Configuration)
   - Theme colors (primaryColor, backgroundColor)
   - Server settings (headless=true, port=8501)

4. **`DASHBOARD_README.md`** (Documentation)
   - Local testing instructions
   - Deployment guide
   - Troubleshooting tips
   - Conference demo flow

---

## 🔧 Technical Implementation Details

### Feature Engineering Function (Critical for Consistency)

```python
FEATURE_COLS = [
    'age', 'bmi', 'children', 'sex', 'smoker', 'region',
    'high_risk', 'smoker_bmi_interaction', 'smoker_age_interaction',
    'cost_complexity_score', 'high_risk_age_interaction',
    'bmi_category', 'age_group', 'family_size'
]

def engineer_features(age, sex, bmi, children, smoker, region):
    # Convert to lowercase
    sex_lower = sex.lower()
    smoker_lower = smoker.lower()
    region_lower = region.lower()

    # BMI categorization (WHO standards)
    if bmi < 18.5:
        bmi_category = 'Underweight'
    elif bmi < 25.0:
        bmi_category = 'Normal'
    elif bmi < 30.0:
        bmi_category = 'Overweight'
    else:
        bmi_category = 'Obese'

    # Age categorization
    if age < 30:
        age_group = '18-29'
    elif age < 40:
        age_group = '30-39'
    elif age < 50:
        age_group = '40-49'
    else:
        age_group = '50-64'

    # Binary conversions
    smoker_binary = 1 if smoker_lower == 'yes' else 0

    # Categorical encoding (MUST match training - alphabetical)
    sex_encoded = 0 if sex_lower == 'female' else 1
    region_encoded = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}.get(region_lower, 0)
    bmi_cat_encoded = {'Normal': 0, 'Obese': 1, 'Overweight': 2, 'Underweight': 3}.get(bmi_category, 0)
    age_grp_encoded = {'18-29': 0, '30-39': 1, '40-49': 2, '50-64': 3}.get(age_group, 0)

    # Compound features
    high_risk = 1 if (smoker_binary == 1 and bmi >= 30) else 0
    smoker_bmi_interaction = smoker_binary * bmi
    smoker_age_interaction = smoker_binary * age
    high_risk_age_interaction = high_risk * age
    family_size = children + 1

    # Cost complexity score
    cost_complexity_score = (
        (smoker_binary * 5) +
        (1 if bmi >= 30 else 0) * 3 +
        (age / 64) * 2 +
        (children * 0.5)
    )

    # Create feature dictionary in exact order
    features = {
        'age': float(age),
        'bmi': float(bmi),
        'children': float(children),
        'sex': float(sex_encoded),
        'smoker': float(smoker_binary),
        'region': float(region_encoded),
        'high_risk': float(high_risk),
        'smoker_bmi_interaction': float(smoker_bmi_interaction),
        'smoker_age_interaction': float(smoker_age_interaction),
        'cost_complexity_score': float(cost_complexity_score),
        'high_risk_age_interaction': float(high_risk_age_interaction),
        'bmi_category': float(bmi_cat_encoded),
        'age_group': float(age_grp_encoded),
        'family_size': float(family_size)
    }

    return pd.DataFrame([features])[FEATURE_COLS]
```

### Training Data Encoding (For SHAP/LIME)

```python
@st.cache_data
def prepare_encoded_training_data(_training_data):
    """
    Encode training data EXACTLY as during training.
    CRITICAL: Must use pd.Categorical().codes for alphabetical encoding.
    """
    X = _training_data[FEATURE_COLS].copy()

    # Encode categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_cols:
        X[col] = pd.Categorical(X[col]).codes

    # Convert all to float
    X = X.astype(float)

    return X
```

**Why This Works:**
- `pd.Categorical().codes` creates codes in **alphabetical order**
- Training data has lowercase values: 'female', 'male' → 'female'=0, 'male'=1 (alphabetical)
- This matches the encoding exactly

---

## 🔄 How to Resume Work / Make Improvements

### If You Need to Update the Dashboard:

1. **Read These Files First:**
```bash
# Core documentation
- PHASE5_DASHBOARD_DEPLOYMENT.md (this file)
- THESIS_COMPLETE_SUMMARY.md (overall project context)
- PHASE4_KNOWLEDGE_SUMMARY.md (SHAP/LIME details)
- DASHBOARD_README.md (deployment guide)

# Implementation files
- app.py (main dashboard code)
- notebooks/00_enhanced_data_preprocessing.py (preprocessing logic)
- notebooks/04d_final_push_0.87.py (model training)
- notebooks/05_shap_global_explanations.py (SHAP implementation)
- notebooks/06_lime_local_explanations.py (LIME implementation)
```

2. **Key Points to Remember:**

**Categorical Encoding:**
- Training uses **lowercase** + `pd.Categorical().codes` (alphabetical)
- Dashboard must match exactly
- Use `prepare_encoded_training_data()` for SHAP/LIME background

**Feature Order:**
- MUST be: age, bmi, children, sex, smoker, region, high_risk, smoker_bmi_interaction, smoker_age_interaction, cost_complexity_score, high_risk_age_interaction, bmi_category, age_group, family_size
- Order is critical - model expects features in this exact sequence

**Model Loading:**
- Model path: `results/models/final_best_model.pkl`
- Training data: `data/processed/insurance_enhanced_processed.csv`
- Use `@st.cache_resource` for model, `@st.cache_data` for data

3. **Testing Changes Locally:**
```bash
# Activate venv
venv\Scripts\activate

# Run Streamlit locally
streamlit run app.py

# Test at http://localhost:8501

# If changes work, commit and push
git add app.py
git commit -m "Update: [describe change]"
git push origin main
# Streamlit Cloud auto-deploys in 1-3 minutes
```

4. **Common Improvements You Might Want:**

**UI Enhancements:**
- Add more visualizations (bar charts, comparison plots)
- Improve color scheme or layout
- Add download buttons (PDF reports, CSV exports)

**Feature Additions:**
- More what-if scenarios (age progression, children impact)
- Historical cost tracking (if longitudinal data available)
- Comparison with multiple patients

**Performance Optimization:**
- Reduce SHAP background samples (100→50) if too slow
- Reduce LIME num_samples (5000→2000) if timeout
- Add progress bars for long computations

**Bug Fixes:**
- Check for edge cases (extreme BMI values)
- Validate input ranges
- Handle model prediction errors gracefully

5. **Debugging Checklist:**

If SHAP/LIME fails:
- [ ] Check categorical encoding matches training
- [ ] Verify feature order is correct
- [ ] Ensure all values are float type
- [ ] Check background data is properly encoded
- [ ] Verify model.predict() works on patient_features

If predictions are wrong:
- [ ] Verify feature engineering matches training
- [ ] Check categorical encoding (alphabetical order)
- [ ] Ensure all 14 features are present
- [ ] Validate BMI/age categorization logic

If deployment fails:
- [ ] Check requirements.txt has all dependencies
- [ ] Verify model file is committed to git
- [ ] Check file paths are relative (not absolute)
- [ ] Review Streamlit Cloud logs for errors

---

## 📊 Performance Benchmarks

### Dashboard Performance:

| Metric | Value | Notes |
|--------|-------|-------|
| Model Load Time | <1 second | Cached with @st.cache_resource |
| Prediction Time | <100ms | Single patient prediction |
| SHAP Computation | ~3-5 seconds | 100 background samples |
| LIME Computation | ~8 seconds | 5,000 perturbations |
| Memory Usage | ~500MB | Streamlit Cloud free tier (1GB limit) |
| Page Load Time | ~2 seconds | Initial cold start |

### Optimization Tips:

**If SHAP is too slow:**
```python
# Reduce background samples
background = shap.sample(X_train, 50, random_state=42)  # Instead of 100
```

**If LIME is too slow:**
```python
# Reduce num_samples
explanation = lime_explainer.explain_instance(
    ...,
    num_samples=2000  # Instead of 5000
)
```

**If memory issues:**
```python
# Don't load full training data, use sample
df_sample = pd.read_csv('...').sample(500, random_state=42)
```

---

## 🎓 Conference Presentation Tips

### 5-Minute Demo Flow:

**Minute 1: Problem & Solution**
- "92% of patients lack cost transparency"
- "We built interpretable AI: R²=0.8770 + SHAP/LIME"

**Minute 2: Live Prediction**
- Enter HIGH-RISK patient (Age=50, BMI=35, Smoker=Yes)
- Show prediction: ~$40,000-$50,000
- Highlight risk category: 🔴 High Risk

**Minute 3: SHAP Explanation**
- Show SHAP waterfall: smoking-BMI dominates
- "Global consistency: same features always important"
- "Scientifically grounded: game theory-based"

**Minute 4: LIME Explanation**
- Show LIME contributions: patient-specific insights
- "8 seconds real-time: patient-friendly"
- "Local accuracy: 82.9% average"

**Minute 5: What-If Impact**
- Toggle smoking cessation: $40,000 → $12,000
- "$28,000 annual savings from quitting!"
- "This motivates behavior change with concrete numbers"

**Questions to Prepare For:**
- Q: Why both SHAP and LIME?
- A: Complementary - SHAP for validation (global), LIME for patients (local, fast)

- Q: How accurate are savings?
- A: Based on R²=0.8770 model with 95% CI, not guarantees but evidence-based estimates

- Q: Can this work in Indonesia?
- A: Methodology transferable, need local data retraining

---

## 📚 References for Future Improvements

### Key Papers to Review:
1. SHAP: Lundberg & Lee (2017) - "A Unified Approach to Interpreting Model Predictions"
2. LIME: Ribeiro et al. (2016) - "Why Should I Trust You?"
3. XGBoost: Chen & Guestrin (2016) - "XGBoost: A Scalable Tree Boosting System"

### Useful Links:
- SHAP Documentation: https://shap.readthedocs.io/
- LIME Documentation: https://lime-ml.readthedocs.io/
- Streamlit Docs: https://docs.streamlit.io/
- Streamlit Cloud: https://share.streamlit.io/

### Advanced Features to Consider:
- SHAP Force Plots (interactive)
- LIME Submodular Pick (diverse explanations)
- Counterfactual explanations (actionable recourse)
- Model monitoring (drift detection)

---

## ✅ Final Checklist

### Completed Today:
- [x] Streamlit dashboard fully implemented (`app.py`)
- [x] Fixed categorical encoding bug (training vs dashboard mismatch)
- [x] Integrated SHAP global explanations
- [x] Integrated LIME local explanations
- [x] Implemented what-if scenario analysis
- [x] Deployed to Streamlit Cloud
- [x] Created IEEE conference paper (`conference-paper.tex`)
- [x] Updated supervisor information (Achmad Udin Zailani)
- [x] Updated README.md (Phase 5 complete)
- [x] Created comprehensive documentation (this file)

### Still Pending (Future Work):
- [ ] Presentation slides preparation
- [ ] Practice conference demo
- [ ] Final thesis chapters completion
- [ ] Thesis defense preparation

---

## 🚀 Quick Command Reference

### Local Development:
```bash
# Run dashboard locally
streamlit run app.py

# Test prediction
python -c "from app import engineer_features; print(engineer_features(30, 'Male', 25.0, 0, 'No', 'Southeast'))"

# Check model
python -c "import pickle; m=pickle.load(open('results/models/final_best_model.pkl','rb')); print(type(m))"
```

### Deployment:
```bash
# Commit changes
git add .
git commit -m "Update dashboard: [description]"
git push origin main
# Streamlit auto-deploys

# Check deployment status
# Visit: https://share.streamlit.io/
```

### Compile Conference Paper:
```bash
# Using LaTeX
pdflatex conference-paper.tex && bibtex conference-paper && pdflatex conference-paper.tex && pdflatex conference-paper.tex

# Or use Overleaf
# Upload to https://www.overleaf.com
```

---

**Document Version:** 1.0
**Created:** October 5, 2025
**Purpose:** Complete reference for Phase 5 dashboard deployment and future improvements
**Live Dashboard:** https://healthcare-cost-predictor.streamlit.app/

---

**END OF PHASE 5 DOCUMENTATION**
