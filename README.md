# Prediksi Biaya Pengobatan Pasien Menggunakan XGBoost dengan Pendekatan Explainable AI

**Student:** Ammar Pavel Zamora Siregar (1202224044)
**Program:** Sarjana Informatika, Universitas Telkom
**Year:** 2025

## Project Overview
This thesis project implements **XGBoost with Explainable AI (SHAP & LIME)** for patient treatment cost prediction using the Kaggle Insurance Cost dataset. The goal is to create transparent, interpretable healthcare cost predictions that empower patients in their decision-making process.

## 🎯 Current Status: Phase 5 COMPLETE - Dashboard Deployed Successfully! 🎉

### 🏆 MAJOR ACHIEVEMENTS:
- ✅ **Phase 3**: Thesis Target R² = 0.8770 ≥ 0.87 ACHIEVED
- ✅ **Phase 4 Step 1**: SHAP Global Explanations IMPLEMENTED
- ✅ **Phase 4 Step 2**: LIME Local Explanations IMPLEMENTED
- ✅ **Phase 4 Step 3**: Streamlit Dashboard DEPLOYED
- ✅ **Phase 5**: Production Deployment COMPLETE
- 🌐 **Live Dashboard**: https://healthcare-cost-predictor.streamlit.app/

---

## 📊 Phase 1: Exploratory Data Analysis (COMPLETED ✅)

### Key Discoveries:
- **🚬 Smoking Status**: Dominant predictor (r=0.787) - smokers pay **280% more** than non-smokers
- **📊 Dataset Quality**: 1,338 records with minimal missing data (0.22%)
- **🔗 Critical Interaction**: BMI × Smoking creates highest cost segment (obese smokers: $41,558 average)
- **📈 Distribution**: Highly right-skewed charges ($1,121 - $63,770) - log transformation applied
- **100% High-Cost Cases**: ALL top 5% expensive cases are smokers (67/67)

### Dataset Characteristics:
- **Source:** Kaggle Insurance Cost Dataset
- **Records:** 1,338 patients
- **Features:** 6 original predictors + 13 engineered features
- **Target:** Medical charges (treatment costs in USD)
- **Missing Values:** Only 3 missing BMI values (0.22%)

### Feature Importance (Correlation with Charges):
1. **Smoker**: 0.787 ⭐ Primary cost driver
2. **Age**: 0.299 📈 Moderate predictor
3. **BMI**: 0.198 📊 Weak but interactive
4. **Children**: 0.068 👶 Minimal impact
5. **Sex**: 0.057 👥 Very weak
6. **Region**: 0.006 🌍 Negligible

---

## 📊 Phase 2: Baseline Models (COMPLETED ✅)

### Enhanced Linear Regression Baseline:
- **R² Score**: 0.8566 (85.66% variance explained)
- **RMSE**: $4,226.08
- **MAE**: $2,332.07
- **Status**: Strong baseline with enhanced preprocessing ✅

### Enhanced Feature Engineering:
| Enhanced Feature | Correlation (r) | Medical Justification |
|------------------|-----------------|------------------------|
| smoker_bmi_interaction | 0.845 | Synergistic smoking-obesity effect |
| high_risk | 0.815 | Compound cardiovascular risk |
| high_risk_age_interaction | 0.799 | Age-amplified high-risk costs |
| smoker_age_interaction | 0.789 | Cumulative smoking damage |
| cost_complexity_score | 0.745 | Healthcare complexity metric |

---

## 📊 Phase 3: XGBoost Optimization & Target Achievement (COMPLETED ✅)

### 3a. Enhanced XGBoost Baseline:
- **R² Score**: 0.8014
- **Status**: Severe overfitting detected (gap = 0.1975) ⚠️
- **Outcome**: Critical need for hyperparameter optimization identified

### 3b. Targeted XGBoost Optimization:
- **R² Score**: 0.8698
- **RMSE**: $4,444.35
- **Status**: Very close to target (gap = 0.0002) ✅
- **Strategy**: RandomizedSearchCV with 150 iterations, proven features focus

### 3c. Final Ensemble Stacking - THESIS TARGET ACHIEVED! 🎉:
- **🏆 R² Score: 0.8770** (87.70% variance explained)
- **💰 RMSE: $4,320**
- **📊 MAE: $2,440**
- **🔥 Best Model**: Stacking_Elastic ensemble with 6 diverse base models
- **✅ Target Status**: R² = 0.8770 ≥ 0.87 with comfortable margin (+0.007)

### Ensemble Configuration:
| Base Model | Type | Role |
|------------|------|------|
| XGBoost_Best | Gradient Boosting | Primary predictor (optimized) |
| XGBoost_Conservative | Gradient Boosting | Stability (high regularization) |
| XGBoost_Aggressive | Gradient Boosting | Pattern capture (low reg) |
| LightGBM | Gradient Boosting | Diversity (alternative algorithm) |
| Ridge Regression | Linear | Bias correction |
| ElasticNet | Linear | Robustness (L1+L2 reg) |
| **Meta-Learner** | **ElasticNet** | **Final aggregation** |

---

## 📊 Phase 4: Explainable AI Implementation (COMPLETED ✅)

### Step 1: SHAP Global Explanations (COMPLETED ✅)

**Implementation:** `05_shap_global_explanations.py`

**Configuration:**
- Explainer Type: PermutationExplainer (model-agnostic)
- Background Sample: 100 data points
- Analysis Sample: 200 predictions
- Computation Time: ~110 seconds

**Key Findings:**

| Rank | Feature | Mean |SHAP| ($) | Interpretation |
|------|---------|-----------------|----------------|
| 1 | smoker_bmi_interaction | 6,397.52 | Smoking-BMI synergy dominates |
| 2 | age | 3,041.68 | Age-related cost increase |
| 3 | high_risk_age_interaction | 1,779.63 | Age amplifies high-risk costs |
| 4 | smoker_age_interaction | 1,388.68 | Cumulative smoking damage |
| 5 | cost_complexity_score | 937.74 | Healthcare complexity metric |

**SHAP Visualizations Generated (9 plots):**
- ✅ SHAP Summary Plot (Beeswarm)
- ✅ SHAP Bar Plot (Global Importance)
- ✅ 3 Waterfall Plots (Low/Medium/High cost patients)
- ✅ 4 Dependence Plots (Top features)

**Critical Insights:**
- **Smoking-BMI Synergy**: $6,397.52 mean impact (2x more than age alone)
- **Base Expected Cost**: $14,120.74 (model's average prediction)
- **High-Cost Patient**: Actual $63,770.43 explained by massive SHAP contributions from smoking-related features
- **Actionable Impact**: ~$8,000 potential savings from smoking cessation

### Step 2: LIME Local Explanations (COMPLETED ✅)

**Implementation:** `06_lime_local_explanations.py`

**Configuration:**
- Explainer Type: LimeTabularExplainer
- Mode: Regression
- Num Features: 10 per explanation
- Num Samples: 5,000 perturbations per patient
- Computation Time: ~8 seconds per patient

**Representative Patient Analysis (5 profiles):**

| Patient Profile | Actual Cost | Predicted Cost | Accuracy | Top Feature Impact |
|-----------------|-------------|----------------|----------|-------------------|
| Low Cost Patient | $1,121.87 | $2,056.05 | 83.2% | bmi: -$18,711.68 |
| Medium Cost Patient | $9,386.16 | $11,986.53 | 72.3% | bmi: -$18,502.36 |
| High Cost Patient | $63,770.43 | $52,451.51 | 82.3% | bmi: +$18,406.46 |
| Young Smoker | $20,167.34 | $16,632.44 | 82.5% | bmi: +$18,575.86 |
| Old Non-Smoker | $12,029.29 | $12,727.17 | 94.2% | bmi: -$18,790.25 |

**LIME Visualizations Generated (7 plots):**
- ✅ 5 Individual patient LIME explanation plots
- ✅ Feature contribution comparison across all patients
- ✅ High cost vs Low cost comparison plot

**LIME vs SHAP Complementarity:**

| Aspect | SHAP | LIME |
|--------|------|------|
| Scope | Global feature importance | Local instance explanation |
| Top Feature | smoker_bmi_interaction | bmi (context-dependent) |
| Avg Impact | $6,397.52 (mean \|SHAP\|) | $18,597.32 (avg \|contrib\|) |
| Speed | ~110s for 200 samples | ~8s per patient |
| Best Use | Model validation, global trends | Patient-facing explanations |

**Patient-Friendly Reports Generated:**
- ✅ 5 Actionable recommendation reports
- ✅ Quantified savings estimates (e.g., $18,400 BMI impact, $15,200 smoking-BMI synergy)
- ✅ Cost drivers and reducers identification
- ✅ **High Cost vs Low Cost Delta**: $74,518 (demonstrates massive lifestyle impact)

**Key Patient Empowerment Insights:**
- **Financial Transparency**: Patients see exact cost drivers with quantified impacts
- **Lifestyle Motivation**: $74,518 delta between high/low cost provides concrete incentive
- **Informed Decision-Making**: Clear, actionable recommendations enable proactive cost management

---

## 📋 Project Phases Status

- [x] **Phase 0:** Environment Setup & GitHub Repository ✅
- [x] **Phase 1:** Data Analysis & EDA ✅
  - [x] Comprehensive exploratory data analysis
  - [x] Feature correlation and interaction analysis
  - [x] Statistical testing and outlier detection
  - [x] Chapter 4 thesis documentation (restructured)
- [x] **Phase 2:** Baseline Models ✅
  - [x] Enhanced Linear Regression (R² = 0.8566)
  - [x] Feature engineering with medical standards
  - [x] Performance evaluation and benchmarking
- [x] **Phase 3:** XGBoost Optimization & Target Achievement ✅
  - [x] Enhanced XGBoost baseline (overfitting detection)
  - [x] Targeted optimization (R² = 0.8698)
  - [x] Final ensemble stacking (R² = 0.8770 ≥ 0.87) 🎉
- [x] **Phase 4:** Explainable AI Implementation ✅
  - [x] **Step 1**: SHAP global explanations (9 visualizations)
  - [x] **Step 2**: LIME local explanations (7 visualizations)
  - [x] **Step 3**: Streamlit dashboard integration
  - [x] **Step 4**: What-if scenario analysis
- [x] **Phase 5:** Dashboard Deployment ✅
  - [x] Streamlit Community Cloud deployment
  - [x] Patient-facing interactive interface
  - [x] Real-time cost prediction with explanations
  - [x] Live URL: https://healthcare-cost-predictor.streamlit.app/
- [x] **Phase 6:** Documentation & Paper Completion ✅
  - [x] Complete Chapter 4 with SHAP & LIME results
  - [x] Methodology documentation
  - [x] Conference paper (IEEE format)
  - [ ] Final thesis submission (Pending)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Git
- Virtual environment

### Installation

```bash
# 1. Clone repository
git clone https://github.com/ammarpvl29/thesis-xgboost-explainable-ai.git
cd thesis-xgboost-explainable-ai

# 2. Setup virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Complete Pipeline

**1. Enhanced Data Preprocessing:**
```bash
python notebooks/00_enhanced_data_preprocessing.py
```
**Output:** Data quality score 10/10, WHO BMI standards integration

**2. Enhanced Linear Regression Baseline:**
```bash
python notebooks/02_enhanced_baseline_linear_regression.py
```
**Output:** R² = 0.8566, enhanced features correlation analysis

**3. Final Ensemble Model (Thesis Target):**
```bash
python notebooks/04d_final_push_0.87.py
```
**Output:** R² = 0.8770 ≥ 0.87 ✅ THESIS TARGET ACHIEVED

**4. SHAP Global Explanations:**
```bash
python notebooks/05_shap_global_explanations.py
```
**Output:** 9 SHAP visualizations, global feature importance analysis

**5. LIME Local Explanations:**
```bash
python notebooks/06_lime_local_explanations.py
```
**Output:** 7 LIME visualizations, patient-specific explanations

---

## 📁 Project Structure

```
thesis-xgboost-explainable-ai/
├── data/
│   ├── raw/                              # Original insurance.csv dataset
│   └── processed/                        # Enhanced processed data ✅
│       ├── insurance_enhanced_processed.csv
│       └── preprocessing_enhanced_summary.json
├── notebooks/
│   ├── 00_enhanced_data_preprocessing.py          # Quality 10/10 ✅
│   ├── 01_data_exploration.py                     # Complete EDA ✅
│   ├── 02_enhanced_baseline_linear_regression.py  # R²=0.8566 ✅
│   ├── 03_enhanced_xgboost_baseline.py            # Overfitting detection ✅
│   ├── 04c_xgboost_targeted_optimization.py       # R²=0.8698 ✅
│   ├── 04d_final_push_0.87.py                     # R²=0.8770 ✅
│   ├── 05_shap_global_explanations.py             # SHAP implementation ✅
│   └── 06_lime_local_explanations.py              # LIME implementation ✅
├── paper/
│   ├── Hasil-Penelitian.tex              # Chapter 4 (restructured) ✅
│   └── Hasil-Penelitian - Copy.tex       # Backup with SHAP/LIME sections
├── results/
│   ├── models/
│   │   ├── final_best_model.pkl          # Ensemble R²=0.8770 ✅
│   │   └── final_optimization_summary.json
│   ├── plots/                            # EDA & model visualizations
│   ├── shap/                             # SHAP analysis results ✅
│   │   ├── shap_global_feature_importance.csv
│   │   ├── shap_interaction_analysis.csv
│   │   └── shap_analysis_summary.json
│   └── lime/                             # LIME analysis results ✅
│       ├── lime_patient_reports.json
│       └── lime_analysis_summary.json
├── src/                                  # Future: utility modules
├── CLAUDE.md                             # AI collaboration guide ✅
├── PHASE4_KNOWLEDGE_SUMMARY.md           # Phase 4 technical summary ✅
└── README.md                             # This file ✅
```

---

## 📊 Complete Results Summary

### Model Performance Evolution:
| Phase | Model | R² Test | RMSE ($) | Status |
|-------|-------|---------|----------|--------|
| Preprocessing | Enhanced Pipeline | - | - | Quality 10/10 |
| Baseline 1 | Enhanced Linear | 0.8566 | 4,226 | Strong baseline ✅ |
| Baseline 2 | XGBoost Default | 0.8014 | 4,974 | Overfitting ⚠️ |
| Optimization | Targeted XGBoost | 0.8698 | 4,444 | Near target ✅ |
| **Final** | **Ensemble Stacking** | **0.8770** | **4,320** | **✅ ACHIEVED** |

### SHAP Global Importance (Top 5):
1. smoker_bmi_interaction: $6,397.52
2. age: $3,041.68
3. high_risk_age_interaction: $1,779.63
4. smoker_age_interaction: $1,388.68
5. cost_complexity_score: $937.74

### LIME Patient Impact Analysis:
- **Average Top Contribution**: $18,597.32
- **High-Cost vs Low-Cost Delta**: $74,518.14
- **Average Prediction Accuracy**: 82.9%
- **Actionable Recommendations**: 5 generated

---

## 🔬 Technical Specifications

### Enhanced Preprocessing:
- **WHO BMI Standards**: Medical categorization (Underweight/Normal/Overweight/Obese)
- **14 Proven Features**: Focused selection avoiding feature bloat
- **Quality Score**: 10.0/10.0 (perfect data quality)

### Final Model Architecture:
- **Type**: StackingRegressor with 6 base models
- **Meta-Learner**: ElasticNet (alpha=1.0, l1_ratio=0.5)
- **Training Time**: ~1.13 seconds
- **Overfitting Gap**: 0.0102 (excellent generalization)

### XAI Implementation:
- **SHAP**: PermutationExplainer (model-agnostic)
- **LIME**: LimeTabularExplainer (5,000 samples/patient)
- **Computation**: Real-time feasible (~8s per patient)

---

## 📚 Key Academic Contributions

### Methodological:
1. **Domain-Informed Preprocessing**: WHO medical standards integration
2. **Systematic Optimization Framework**: Baseline → Diagnosis → Optimization → Ensemble
3. **Feature Engineering**: Interaction features (smoker_bmi_interaction r=0.845)
4. **XAI Readiness**: Dual SHAP/LIME framework for comprehensive interpretability

### Empirical:
1. **Smoking Impact Quantification**: 280% cost differential validated
2. **Synergy Effect Measurement**: 370% increase for obese smokers
3. **Benchmark Performance**: R² = 0.8770 for small datasets
4. **XAI Validation**: SHAP-LIME complementarity demonstrated

### Practical:
1. **Patient Empowerment**: Quantified savings estimates (~$8,000 smoking cessation)
2. **Wellness Program ROI**: $23,600/smoker/year potential savings
3. **Risk Stratification**: high_risk indicator for targeted interventions
4. **Production-Ready**: Fast computation, excellent generalization

---

## 📖 Documentation

### Thesis Documentation:
- **Chapter 4**: `paper/Hasil-Penelitian.tex` (Results & Discussion format)
- **Technical Guide**: `CLAUDE.md` (Project conventions & setup)
- **Phase 4 Summary**: `PHASE4_KNOWLEDGE_SUMMARY.md`

### Academic Standards:
- LaTeX thesis format (proper Chapter 4 structure)
- Complete methodology documentation
- Reproducible research (all scripts documented)

---

## 🌐 Live Dashboard

**🚀 Try the Interactive Dashboard:**
- **URL**: https://healthcare-cost-predictor.streamlit.app/
- **Features**:
  - ✅ Real-time cost prediction with confidence intervals
  - ✅ SHAP global feature importance analysis
  - ✅ LIME patient-specific explanations (~8 seconds)
  - ✅ What-if scenario analysis (smoking cessation, weight management)
  - ✅ Risk categorization (Low/Medium/High)
  - ✅ Comparison with population/smoker averages

**Dashboard Screenshots:**
- Patient input form with WHO BMI categorization
- Interactive SHAP waterfall plots
- LIME feature contribution analysis
- What-if savings calculator ($8,000-$45,200 potential)

---

## 📄 Conference Paper

**IEEE Conference Paper**: `conference-paper.tex`

**Paper Details:**
- **Title**: Interpretable Healthcare Cost Prediction Using XGBoost with Dual Explainable AI Framework
- **Authors**:
  - Ammar Pavel Zamora Siregar
  - Achmad Udin Zailani, S.Kom., M.Kom.
  - Nurul Ilmi, S.Kom, M.T
- **Institution**: Universitas Telkom, School of Informatics
- **Format**: IEEE Conference Template (A4)

**Compile Paper:**
```bash
pdflatex conference-paper.tex
bibtex conference-paper
pdflatex conference-paper.tex
pdflatex conference-paper.tex
```

Or use **Overleaf** (online LaTeX editor) for easier compilation.

---

## 🔄 Next Steps

### Conference Presentation
- [x] Dashboard deployed and accessible
- [x] Conference paper completed
- [ ] Presentation slides preparation
- [ ] Practice demo flow (5-minute presentation)

### Final Thesis Submission
- [ ] Complete all thesis chapters
- [ ] Abstract and conclusions
- [ ] Final thesis defense preparation
- [ ] Submission to university repository

---

## 📚 Dependencies

```
# Core Data Science
pandas>=1.5.0
numpy>=1.24.0
scipy>=1.10.0

# Machine Learning
scikit-learn>=1.3.0
xgboost>=1.7.0
lightgbm>=3.3.0

# Explainable AI
shap>=0.41.0          # ✅ Implemented
lime>=0.2.0           # ✅ Implemented

# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.14.0

# Dashboard (Phase 4 Step 3)
streamlit>=1.22.0     # ✅ Implemented
```

---

## 👨‍🎓 About This Thesis

This research contributes to healthcare AI transparency by combining:
- **Advanced ML**: XGBoost ensemble achieving R² = 0.8770
- **Explainable AI**: Dual SHAP/LIME framework for comprehensive interpretability
- **Patient Empowerment**: Quantified cost insights ($74,518 lifestyle impact delta)
- **Production-Ready**: Fast computation (~8s per explanation), excellent generalization

**University**: Universitas Telkom, School of Informatics
**Student**: Ammar Pavel Zamora Siregar (1202224044)
**Supervisors**:
- Achmad Udin Zailani, S.Kom., M.Kom.
- Nurul Ilmi, S.Kom, M.T
**Year**: 2025

---

## 📧 Contact

For questions or collaboration inquiries regarding this thesis project:
- **Student**: Ammar Pavel Zamora Siregar
- **Email**: ammarpvl@student.telkomuniversity.ac.id
- **GitHub**: (https://github.com/ammarpvl29/thesis-xgboost-explainable-ai.git)

---

## 📄 License

This thesis project is for academic purposes. Dataset used is publicly available from Kaggle Insurance Cost dataset.

---

**Last Updated**: October 5, 2025
**Status**: Phase 5 Dashboard Deployment Complete ✅
**Live Dashboard**: https://healthcare-cost-predictor.streamlit.app/
**Conference Paper**: Ready for submission 📄
