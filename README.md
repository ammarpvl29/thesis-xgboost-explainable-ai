# Prediksi Biaya Pengobatan Pasien Menggunakan XGBoost dengan Pendekatan Explainable AI

**Student:** Ammar Pavel Zamora Siregar (1202224044)  
**Program:** Sarjana Informatika, Universitas Telkom  
**Year:** 2025

## Project Overview
This thesis project implements **XGBoost with Explainable AI (SHAP & LIME)** for patient treatment cost prediction using the Kaggle Insurance Cost dataset. The goal is to create transparent, interpretable healthcare cost predictions that empower patients in their decision-making process.

## 🎯 Current Status: Phase 2 - Baseline Model Completed ✅

### Phase 1 Key Discoveries:
- **🚬 Smoking Status**: Dominant predictor (r=0.787) - smokers pay **280% more** than non-smokers
- **📊 Dataset Quality**: 1,338 records with minimal missing data (0.22%)
- **🔗 Critical Interaction**: BMI × Smoking creates highest cost segment (obese smokers: $41,558 average)
- **📈 Distribution**: Highly right-skewed charges ($1,121 - $63,770) - log transformation needed

### Phase 2 Baseline Results ✅:
- **🎯 R² Score: 0.8637** (86.37% variance explained) - **EXCEEDS THESIS TARGET >0.85!**
- **💰 RMSE: $4,120.52** - Excellent prediction accuracy
- **📊 MAE: $2,260.53** - Strong average prediction performance
- **🔍 Top Predictors**: high_risk (coef: 6,353), smoker (coef: 5,274), age (coef: 4,061)
## 📊 Dataset Characteristics
- **Source:** Kaggle Insurance Cost Dataset
- **Records:** 1,338 patients
- **Features:** 6 predictors (age, sex, bmi, children, smoker, region) + 1 target (charges)
- **Target:** Medical charges (treatment costs in USD)
- **Missing Values:** Only 3 missing BMI values (0.22%)

### Feature Importance (Correlation with Charges):
1. **Smoker**: 0.787 ⭐ Primary cost driver
2. **Age**: 0.299 📈 Moderate predictor
3. **BMI**: 0.198 📊 Weak but interactive
4. **Children**: 0.068 👶 Minimal impact
5. **Sex**: 0.057 👥 Very weak
6. **Region**: 0.006 🌍 Negligible

## 📋 Project Phases
- [x] **Phase 0:** Environment Setup & GitHub Repository ✅
- [x] **Phase 1:** Data Analysis & EDA ✅ **(COMPLETED)**
  - [x] Comprehensive exploratory data analysis
  - [x] Feature correlation and interaction analysis  
  - [x] Statistical testing and outlier detection
  - [x] Feature engineering and data preprocessing
  - [x] Chapter 4 thesis documentation
- [x] **Phase 2:** Baseline Linear Regression ✅ **(COMPLETED)**
  - [x] Algorithm 2 implementation with R² = 0.8637
  - [x] Feature importance analysis (17 engineered features)
  - [x] Performance evaluation exceeding thesis targets
  - [x] Baseline benchmark established for XGBoost comparison
- [ ] **Phase 3:** XGBoost Implementation 🔄 **(NEXT)**
- [ ] **Phase 4:** Explainable AI Integration (SHAP & LIME)
- [ ] **Phase 5:** Dashboard Development
- [ ] **Phase 6:** Documentation & Paper Completion

## 🚀 Quick Start

### Running the Baseline Model
```bash
# Run baseline Linear Regression implementation
python notebooks/02_baseline_linear_regression.py
```

**Expected Outcome:**
- ✅ **R² Score: 0.8637** (86.37% variance explained) - exceeds thesis target
- ✅ **RMSE: $4,120.52** with strong prediction accuracy
- ✅ **Feature importance analysis** confirming smoking dominance
- ✅ **Model artifacts saved** to `results/models/baseline_model_summary.json`
- ✅ **Visualizations generated**: feature importance & prediction evaluation plots
cd thesis-xgboost-explainable-ai
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies  
pip install pandas numpy matplotlib seaborn scipy

# 3. Run EDA analysis
python notebooks/01_data_exploration.py
```

### Setup Instructions
1. **Prerequisites:**
   - Python 3.11+
   - Git

2. **Clone Repository:**
   ```bash
   git clone https://github.com/ammarpvl29/thesis-xgboost-explainable-ai.git
   cd thesis-xgboost-explainable-ai
   ```

3. **Environment Setup:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

4. **Dataset Setup:**
   - Insurance dataset (`insurance.csv`) is included in `data/raw/`
   - Processed data available in `data/processed/`

5. **Run Analysis:**
   ```bash
   # Option 1: Run complete EDA
   python notebooks/01_data_exploration.py
   
   # Option 2: Interactive exploration  
   jupyter notebook notebooks/
   ```

## 📁 Project Structure
```
thesis-xgboost-explainable-ai/
├── data/
│   ├── raw/                    # Original insurance.csv dataset
│   └── processed/              # Feature-engineered data ✅
├── notebooks/
│   ├── 01_data_exploration.py  # Complete EDA analysis ✅
│   └── 02_baseline_linear_regression.py # Baseline model (R²=0.8637) ✅
├── paper/
│   └── Hasil-Penelitian.tex    # Chapter 4 with baseline results ✅
├── results/
│   ├── plots/                  # Generated visualizations ✅
│   └── models/                 # Model artifacts & summaries ✅
├── src/                        # Future: XGBoost & XAI modules
├── CLAUDE.md                   # Project documentation ✅
└── README.md                   # This file ✅
```

## 📊 Key Findings Summary

### 🔍 EDA Results:
- **Most Important Discovery**: Smoking status completely dominates healthcare costs
- **Cost Impact**: Average smoker pays **$32,050** vs non-smoker **$8,434** (280% difference)
- **High-Cost Cases**: 100% of top 5% most expensive cases are smokers (67/67)
- **Critical Interaction**: BMI × Smoking multiplier effect (obese smokers: $41,558)

### 🎯 Baseline Model Performance:
- **R² Score: 0.8637** (86.37% variance explained) - **EXCEEDS THESIS TARGET >0.85**
- **Top Features**: high_risk (coef: 6,353), smoker (coef: 5,274), age (coef: 4,061)
- **Prediction Accuracy**: RMSE $4,120.52, MAE $2,260.53, MAPE 26.03%
- **Cross-Validation**: Stable R² = 0.8603 (±0.0867) across 5 folds

### 🎯 XGBoost Target:
- **Target Performance**: R² > 0.87 to show meaningful improvement over baseline
- **XAI Potential**: Clear feature hierarchy will create consistent, actionable explanations
- **Patient Focus**: Lifestyle-based cost drivers enable meaningful interventions

## 🔬 Technical Details

### Dataset Statistics:
- **Sample Size**: 1,338 patients
- **Age Range**: 18-64 years (mean: 39.2)
- **BMI Range**: 15.96-53.13 (mean: 30.7) 
- **Cost Range**: $1,122-$63,770 (mean: $13,270)
- **Demographics**: Balanced sex (50.5% male) and regions (24-27% each)

### Data Quality:
- **Completeness**: 99.78% (only 3 missing BMI values)
- **Outliers**: 10.4% of cases using IQR method
- **Distribution**: Right-skewed target (skewness: 1.516)

## 📖 Academic Documentation
- **Chapter 4**: Complete results and discussion in `paper/Hasil-Penelitian.tex`
- **Methodology**: Comprehensive EDA following academic standards
- **Visualizations**: Statistical plots saved in `results/plots/`

## 🔄 Next Steps (Phase 3)
- [ ] Implement XGBoost core model (Algorithm 3)
- [ ] Hyperparameter optimization with RandomizedSearchCV
- [ ] Performance comparison: XGBoost vs Baseline (target R² > 0.87)
- [ ] Prepare for SHAP/LIME integration in Phase 4

## 📚 Dependencies
```
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.10.0
scikit-learn  # Coming in Phase 2
xgboost      # Coming in Phase 2
shap         # Coming in Phase 3
lime         # Coming in Phase 3
```

## 👨‍🎓 About This Thesis
This research contributes to healthcare AI transparency by combining:
- **Advanced ML**: XGBoost for accurate cost prediction
- **Explainable AI**: SHAP & LIME for model interpretability
- **Patient Empowerment**: User-friendly explanations for medical cost decisions

**University**: Universitas Telkom, Fakultas Informatika  
**Thesis Advisor**: [To be updated]  
**Expected Completion**: 2025
