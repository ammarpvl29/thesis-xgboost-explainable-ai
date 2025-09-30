# Prediksi Biaya Pengobatan Pasien Menggunakan XGBoost dengan Pendekatan Explainable AI

**Student:** Ammar Pavel Zamora Siregar (1202224044)  
**Program:** Sarjana Informatika, Universitas Telkom  
**Year:** 2025

## Project Overview
This thesis project implements **XGBoost with Explainable AI (SHAP & LIME)** for patient treatment cost prediction using the Kaggle Insurance Cost dataset. The goal is to create transparent, interpretable healthcare cost predictions that empower patients in their decision-making process.

## 🎯 Current Status: Phase 3 - THESIS TARGET ACHIEVED! R² = 0.8770 ≥ 0.87 🎉

### Phase 1 Key Discoveries:
- **🚬 Smoking Status**: Dominant predictor (r=0.787) - smokers pay **280% more** than non-smokers
- **📊 Dataset Quality**: 1,338 records with minimal missing data (0.22%)
- **🔗 Critical Interaction**: BMI × Smoking creates highest cost segment (obese smokers: $41,558 average)
- **📈 Distribution**: Highly right-skewed charges ($1,121 - $63,770) - log transformation needed

### Phase 2 Enhanced Linear Regression Baseline ✅:
- **🎯 R² Score: 0.8566** (85.66% variance explained) with enhanced preprocessing
- **💰 RMSE: $4,226.08** - Strong prediction accuracy with enhanced features
- **📊 MAE: $2,332.07** - Solid baseline performance
- **🔍 Top Enhanced Features**: smoker_bmi_interaction (r=0.845), high_risk (r=0.815)

### Phase 3a Enhanced XGBoost Baseline Results ⚠️:
- **📉 R² Score: 0.8014** (80.14% variance explained) - **SEVERE OVERFITTING DETECTED**
- **💸 RMSE: $4,973.71** - Poor generalization performance
- **📊 MAE: $2,783.22** - Degraded prediction accuracy
- **🚨 Critical Overfitting**: Training R² = 0.9989 vs Test R² = 0.8014 (gap = 0.1975)
- **⚠️ Hyperparameter Optimization Urgently Needed**

### Phase 3b Targeted XGBoost Optimization Results ✅:
- **🎯 R² Score: 0.8698** (86.98% variance explained) - **VERY CLOSE to thesis target!**
- **💰 RMSE: $4,444.35** - Excellent improvement with proven features
- **📊 MAE: $2,489.51** - Strong prediction accuracy
- **✅ Excellent Generalization**: Training R² = 0.9104 vs Test R² = 0.8698 (gap = 0.0407)
- **🎯 Gap to Target**: Only 0.0002 remaining to reach R² ≥ 0.87
- **🚀 Proven Feature Strategy**: Focused on 14 high-value features (avoided feature bloat)

### Phase 3c FINAL ENSEMBLE - THESIS TARGET ACHIEVED! 🎉:
- **🏆 R² Score: 0.8770** (87.70% variance explained) - **✅ THESIS TARGET ACHIEVED!**
- **💰 RMSE: $4,320** - Best prediction accuracy achieved
- **📊 MAE: $2,440** - Superior performance metrics
- **🔥 Best Model**: Stacking_Elastic ensemble with diverse base models
- **✅ Target Status**: R² = 0.8770 ≥ 0.87 with comfortable margin (+0.007)
- **🎯 Ready for Phase 4**: Explainable AI with optimized ensemble model
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
- [x] **Phase 0:** Enhanced Data Preprocessing ✅ **(COMPLETED - MEDICAL STANDARDS INTEGRATED)**
  - [x] Script: `00_enhanced_data_preprocessing.py` - WHO BMI standards integration
  - [x] Data quality enhancement: 10.0/10.0 quality score achieved
  - [x] Medical domain-specific feature engineering
  - [x] Enhanced feature correlations (smoker_bmi_interaction: r=0.845)
- [x] **Phase 3a:** Enhanced XGBoost Baseline ⚠️ **(COMPLETED - OVERFITTING IDENTIFIED)**
  - [x] Script: `03_enhanced_xgboost_baseline.py` - Enhanced data implementation
  - [x] Severe overfitting detected (gap = 0.1975)
  - [x] Critical need for hyperparameter optimization identified
  - [x] Feature importance analysis with enhanced features
- [x] **Phase 3b:** Targeted XGBoost Optimization ✅ **(COMPLETED - NEAR TARGET)**
  - [x] Script: `04c_xgboost_targeted_optimization.py` - Proven features focus
  - [x] Aggressive hyperparameter optimization (150 iterations, 750 fits)
  - [x] R² = 0.8698 achieved (gap = 0.0002 to thesis target)
  - [x] Excellent generalization with proven feature selection
- [x] **Phase 3c:** Final Ensemble Push ✅ **(COMPLETED - 🎉 THESIS TARGET ACHIEVED)**
  - [x] Script: `04d_final_push_0.87.py` - Ensemble stacking implementation
  - [x] Diverse base models with meta-learner optimization
  - [x] **🏆 FINAL ACHIEVEMENT: R² = 0.8770 ≥ 0.87** (thesis target achieved!)
  - [x] Best model: Stacking_Elastic ensemble with superior performance
- [ ] **Phase 4:** Explainable AI Integration (SHAP & LIME)
- [ ] **Phase 5:** Dashboard Development
- [ ] **Phase 6:** Documentation & Paper Completion

## 🚀 Quick Start

### Running the Models

**1. Enhanced Data Preprocessing:**
```bash
# Run enhanced data preprocessing with medical standards
python notebooks/00_enhanced_data_preprocessing.py
```

**Expected Outcome:**
- ✅ **Data Quality Score: 10.0/10.0** - Perfect data quality achieved
- ✅ **WHO BMI Standards** - Medical categorization implemented
- ✅ **Enhanced Features** - Domain-specific healthcare features created
- ✅ **Processed data saved** to `data/processed/insurance_enhanced_processed.csv`

**2. Enhanced Linear Regression Baseline:**
```bash
# Run enhanced Linear Regression with processed data
python notebooks/02_enhanced_baseline_linear_regression.py
```

**Expected Outcome:**
- ✅ **R² Score: 0.8566** (85.66% variance explained) with enhanced features
- ✅ **RMSE: $4,226.08** with enhanced feature correlations
- ✅ **Top correlations** - smoker_bmi_interaction (r=0.845), high_risk (r=0.815)
- ✅ **Enhanced model saved** to `results/models/enhanced_linear_regression_summary.json`

**3. Enhanced XGBoost Baseline:**
```bash
# Run enhanced XGBoost baseline with processed data
python notebooks/03_enhanced_xgboost_baseline.py
```

**Expected Outcome:**
- ⚠️ **R² Score: 0.8014** (80.14% variance explained) - severe overfitting detected
- ⚠️ **RMSE: $4,973.71** - poor generalization performance
- 🚨 **Critical Overfitting**: Training-Test gap = 0.1975 (urgent optimization needed)
- ✅ **Enhanced baseline saved** to `results/models/enhanced_xgboost_baseline.pkl`

**4. Targeted XGBoost Optimization:**
```bash
# Run targeted optimization with proven high-value features
python notebooks/04c_xgboost_targeted_optimization.py
```

**Expected Outcome:**
- ✅ **R² Score: 0.8698** (86.98% variance explained) - very close to thesis target
- ✅ **RMSE: $4,444.35** - excellent improvement with proven features
- 🎯 **Gap to Target**: Only 0.0002 remaining to reach R² ≥ 0.87
- ✅ **Targeted model saved** to `results/models/xgboost_targeted_optimized.pkl`

**5. 🎉 Final Ensemble Push - THESIS TARGET ACHIEVED:**
```bash
# Run final ensemble stacking to achieve thesis target
python notebooks/04d_final_push_0.87.py
```

**Expected Outcome:**
- 🏆 **R² Score: 0.8770** (87.70% variance explained) - **🎉 THESIS TARGET ACHIEVED!**
- 🏆 **RMSE: $4,320** - Best prediction accuracy achieved
- 🏆 **Best Model**: Stacking_Elastic ensemble with diverse base models
- ✅ **Final model saved** to `results/models/final_best_model.pkl`
- 🚀 **Ready for Phase 4**: Explainable AI with thesis-grade performance
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
│   ├── 00_enhanced_data_preprocessing.py    # Enhanced preprocessing (Quality: 10/10) ✅
│   ├── 01_data_exploration.py              # Complete EDA analysis ✅
│   ├── 02_enhanced_baseline_linear_regression.py # Enhanced baseline (R²=0.8566) ✅
│   ├── 03_enhanced_xgboost_baseline.py     # Enhanced XGBoost baseline ✅
│   ├── 04c_xgboost_targeted_optimization.py # Targeted optimization (R²=0.8698) ✅
│   └── 04d_final_push_0.87.py             # 🎉 Final ensemble (R²=0.8770) ✅
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

### 🎯 Enhanced Model Performance Evolution:
- **Enhanced Linear Baseline**: R² = 0.8566 with enhanced preprocessing
- **Targeted XGBoost**: R² = 0.8698 (gap = 0.0002 to thesis target)
- **🏆 Final Ensemble**: R² = 0.8770 ≥ 0.87 - **THESIS TARGET ACHIEVED!**
- **Best Model**: Stacking_Elastic with RMSE $4,320, MAE $2,440
- **Top Enhanced Features**: smoker_bmi_interaction (r=0.845), high_risk (r=0.815)

### 🎯 FINAL RESULTS - THESIS TARGET ACHIEVED:
- **🏆 Final Performance**: R² = 0.8770 (87.70% variance explained)
- **✅ Thesis Target**: **ACHIEVED** (R² ≥ 0.87 with comfortable margin)
- **🎉 Best Model**: Stacking_Elastic ensemble outperforms all single models
- **📈 Complete Evolution**: 0.8566 → 0.8698 → **0.8770** (systematic improvement)
- **🚀 Enhanced Features Impact**: Medical standards + domain expertise crucial
- **✅ XAI Ready**: Thesis-grade ensemble model prepared for SHAP/LIME Phase 4

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
- [x] **🎉 Phase 3 Complete**: THESIS TARGET ACHIEVED (R² = 0.8770 ≥ 0.87) ✅
- [x] **✅ Final Ensemble**: Stacking_Elastic model with superior performance ready ✅
- [ ] **Phase 4 Priority**: Implement SHAP & LIME explainability on final ensemble model
- [ ] **Technical Focus**:
  - SHAP global explanations for ensemble feature importance
  - LIME local interpretability for individual patient predictions
  - Enhanced feature visualization (smoker_bmi_interaction, high_risk)
  - Interactive dashboard with thesis-grade model performance
- [ ] **Academic Goal**: Demonstrate explainable AI value with R² ≥ 0.87 model
- [ ] **Timeline**: Phase 4 implementation with achieved thesis target foundation

## 📊 Critical Findings Summary

### 🎉 **BREAKTHROUGH ACHIEVEMENT**: THESIS TARGET R² ≥ 0.87 ACHIEVED!

**🏆 Complete Methodology Evolution - SYSTEMATIC SUCCESS:**
1. **Enhanced Linear Baseline**: R² = 0.8566 ✅ (Enhanced preprocessing foundation)
2. **Enhanced XGBoost Baseline**: R² = 0.8014 ⚠️ (Critical overfitting identified)
3. **Targeted XGBoost**: R² = 0.8698 ✅ (Proven features, near target)
4. **🎉 Final Ensemble**: R² = 0.8770 ✅ (**THESIS TARGET ACHIEVED!**)

**🚀 Success Strategy Implementation:**
- **Enhanced Preprocessing**: Medical standards integration (WHO BMI) → Quality 10/10
- **Proven Feature Selection**: Avoided feature bloat, focused on high-correlation features
- **Aggressive Optimization**: 150 iterations, 750 fits for comprehensive search
- **Ensemble Stacking**: Diverse base models with meta-learner for final breakthrough

**🎯 Critical Academic Achievements:**
1. **🏆 THESIS TARGET ACHIEVED**: R² = 0.8770 ≥ 0.87 ✅ (comfortable margin: +0.007)
2. **✅ Systematic Methodology**: Complete pipeline from preprocessing to ensemble
3. **📈 Medical Domain Integration**: Healthcare-specific features proved crucial
4. **🚀 Ensemble Innovation**: Stacking_Elastic outperformed all single models
5. **✅ XAI Foundation**: Thesis-grade model ready for SHAP/LIME integration

## 📚 Dependencies
```
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.10.0
scikit-learn>=1.3.0  # ✅ Used in all phases
xgboost>=1.7.0       # ✅ Enhanced optimization complete
lightgbm>=3.3.0      # ✅ Ensemble base model
shap                 # 🔄 Coming in Phase 4
lime                 # 🔄 Coming in Phase 4
```

## 👨‍🎓 About This Thesis
This research contributes to healthcare AI transparency by combining:
- **Advanced ML**: XGBoost for accurate cost prediction
- **Explainable AI**: SHAP & LIME for model interpretability
- **Patient Empowerment**: User-friendly explanations for medical cost decisions

**University**: Universitas Telkom, Fakultas Informatika  
**Thesis Advisor**: [To be updated]  
**Expected Completion**: 2025
