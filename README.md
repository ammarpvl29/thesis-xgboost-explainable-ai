# Prediksi Biaya Pengobatan Pasien Menggunakan XGBoost dengan Pendekatan Explainable AI

**Student:** Ammar Pavel Zamora Siregar (1202224044)  
**Program:** Sarjana Informatika, Universitas Telkom  
**Year:** 2025

## Project Overview
This thesis project implements **XGBoost with Explainable AI (SHAP & LIME)** for patient treatment cost prediction using the Kaggle Insurance Cost dataset. The goal is to create transparent, interpretable healthcare cost predictions that empower patients in their decision-making process.

## 🎯 Current Status: Phase 3b - XGBoost Optimization Complete ✅

### Phase 1 Key Discoveries:
- **🚬 Smoking Status**: Dominant predictor (r=0.787) - smokers pay **280% more** than non-smokers
- **📊 Dataset Quality**: 1,338 records with minimal missing data (0.22%)
- **🔗 Critical Interaction**: BMI × Smoking creates highest cost segment (obese smokers: $41,558 average)
- **📈 Distribution**: Highly right-skewed charges ($1,121 - $63,770) - log transformation needed

### Phase 2 Linear Regression Baseline ✅:
- **🎯 R² Score: 0.8637** (86.37% variance explained) - **EXCEEDS THESIS TARGET >0.85!**
- **💰 RMSE: $4,120.52** - Excellent prediction accuracy
- **📊 MAE: $2,260.53** - Strong average prediction performance
- **🔍 Top Predictors**: high_risk (coef: 6,353), smoker (coef: 5,274), age (coef: 4,061)

### Phase 3a XGBoost Baseline Results ⚠️:
- **📉 R² Score: 0.8309** (83.09% variance explained) - **BELOW Linear Regression baseline!**
- **💸 RMSE: $4,589.52** - Worse than Linear Regression (+11.4%)
- **📊 MAE: $2,565.82** - Degraded performance (+13.5%)
- **🚨 Overfitting Signs**: Training R² = 0.9747 vs Test R² = 0.8309 (huge gap)
- **⚠️ Professor's Target**: R² = 0.8309 < 0.86 (target missed)

### Phase 3b XGBoost Enhanced Optimization Results ✅:
- **🎯 R² Score: 0.8618** (86.18% variance explained) - **MEETS Professor's Target >0.86!**
- **💰 RMSE: $4,145.12** - Significant improvement over baseline (-9.7%)
- **📊 MAE: $2,384.88** - Better prediction accuracy (-7.0%)
- **✅ Generalization**: Training R² = 0.8783 vs Test R² = 0.8618 (healthy gap: 0.0165)
- **🚀 Feature Engineering**: 5 interaction features added (smoker×BMI, smoker×age, etc.)
- **⚙️ Optimal Parameters**: n_estimators=1000, max_depth=4, learning_rate=0.05, subsample=0.9
- **🔑 Key Achievement**: **Professor's target achieved** but thesis target (>0.87) still pending
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
- [x] **Phase 3a:** XGBoost Baseline ⚠️ **(COMPLETED - BASELINE ESTABLISHED)**
  - [x] XGBoost baseline implementation (R² = 0.8309)
  - [x] Performance analysis showing need for hyperparameter tuning
  - [x] Overfitting identification and diagnosis
  - [x] Feature importance comparison with Linear Regression
- [x] **Phase 3b:** XGBoost Enhanced Optimization ✅ **(COMPLETED - PROFESSOR'S TARGET ACHIEVED)**
  - [x] Feature interaction engineering (5 key interactions added)
  - [x] Hyperparameter optimization with RandomizedSearchCV (400 iterations)
  - [x] Enhanced model performance: R² = 0.8618 (meets Professor's target >0.86)
  - [x] Overfitting resolution: Training-Test gap reduced to 0.0165
  - [x] Cross-validation stability: CV R² = 0.8568 (±0.0089)
- [ ] **Phase 4:** Explainable AI Integration (SHAP & LIME)
- [ ] **Phase 5:** Dashboard Development
- [ ] **Phase 6:** Documentation & Paper Completion

## 🚀 Quick Start

### Running the Models

**1. Linear Regression Baseline:**
```bash
# Run baseline Linear Regression implementation
python notebooks/02_baseline_linear_regression.py
```

**Expected Outcome:**
- ✅ **R² Score: 0.8637** (86.37% variance explained) - exceeds thesis target
- ✅ **RMSE: $4,120.52** with strong prediction accuracy
- ✅ **Feature importance analysis** confirming smoking dominance
- ✅ **Model artifacts saved** to `results/models/baseline_model_summary.json`

**2. XGBoost Baseline:**
```bash
# Run XGBoost baseline (without optimization)
python notebooks/03_xgboost_implementation.py
```

**Expected Outcome:**
- ⚠️ **R² Score: 0.8309** (83.09% variance explained) - below Linear Regression
- ⚠️ **RMSE: $4,589.52** - needs hyperparameter optimization
- 🔍 **Overfitting diagnosis**: Training vs Test performance gap identified
- ✅ **Model artifacts saved** to `results/models/xgboost_baseline_model.pkl`

**3. XGBoost Enhanced Optimization:**
```bash
# Run XGBoost with hyperparameter optimization and feature interactions
python notebooks/04_xgboost_optimization.py
```

**Expected Outcome:**
- ✅ **R² Score: 0.8618** (86.18% variance explained) - **MEETS Professor's target >0.86**
- ✅ **RMSE: $4,145.12** - Significant improvement over baseline (-9.7%)
- 🎯 **Feature Interactions**: 5 key interaction features enhance predictive power
- ✅ **Model artifacts saved** to `results/models/xgboost_optimized_summary.json`
- 🚀 **Ready for Phase 4**: SHAP/LIME integration with optimized model
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

### 🎯 XGBoost Enhanced Results:
- **Current Performance**: R² = 0.8618 (86.18% variance explained)
- **Professor's Target**: ✅ **ACHIEVED** (R² > 0.86)
- **Thesis Target**: ⚠️ **PENDING** (R² > 0.87 - only 0.49% short)
- **Linear Regression Comparison**: Still behind by 0.19% (0.8637 vs 0.8618)
- **Feature Interactions Impact**: 5 engineered interactions boosted performance by 3.09%
- **XAI Readiness**: Optimized model prepared for SHAP/LIME integration in Phase 4

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

## 🔄 Next Steps (Phase 4 - Explainable AI Integration)
- [x] **Phase 3b Complete**: XGBoost optimization achieved Professor's target (R² = 0.8618 > 0.86) ✅
- [ ] **Phase 4 Priority**: Implement SHAP & LIME explainability on optimized XGBoost model
- [ ] **Technical Focus**:
  - SHAP global and local explanations for feature interactions
  - LIME local interpretability for individual predictions
  - Feature importance visualization comparing Linear vs XGBoost
  - Interactive dashboard for patient cost explanations
- [ ] **Academic Goal**: Demonstrate explainable AI value for healthcare decision-making
- [ ] **Timeline**: Target completion by October 15 for thesis defense preparation

## 📊 Critical Findings Summary

### 🚨 **Key Insight**: XGBoost Optimization Success & Feature Interactions Impact

**Complete Performance Evolution Analysis:**
1. **Linear Regression Baseline**: R² = 0.8637 ✅ (Strong performance benchmark)
2. **XGBoost Baseline**: R² = 0.8309 ❌ (-3.8% regression from Linear)
3. **XGBoost Enhanced**: R² = 0.8618 ✅ (+3.09% improvement, meets Professor's target)

**Optimization Strategy Success:**
- **Overfitting Resolution**: Training-Test gap reduced from 0.144 to 0.0165
- **Feature Engineering Impact**: 5 interaction features crucial for performance boost
- **Parameter Optimization**: 400-iteration RandomizedSearchCV found optimal configuration
- **Generalization**: Excellent cross-validation stability (R² = 0.8568 ± 0.0089)

**Critical Academic Insights:**
1. **Professor's Target Achieved**: R² = 0.8618 > 0.86 ✅
2. **Thesis Target Pending**: Missing R² > 0.87 by only 0.49% (very close)
3. **Linear vs XGBoost**: XGBoost still behind Linear by 0.19% (marginal difference)
4. **Feature Interactions**: Smoking×BMI and Smoking×Age interactions are key predictors
5. **Ready for XAI**: Optimized model prepared for SHAP/LIME integration

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
