# CLAUDE.md - XGBoost Explainable AI Thesis Project

## Project Overview

This repository contains a thesis project implementing **XGBoost with Explainable AI (SHAP & LIME)** for patient treatment cost prediction using the Kaggle Insurance Cost dataset.

**Author:** Ammar Pavel Zamora Siregar (1202224044)  
**Institution:** Universitas Telkom, Sarjana Informatika  
**Year:** 2025

## Environment Setup

### Python Environment
- **Python Version:** 3.11+ (recommended via pyenv)
- **Virtual Environment:** Standard Python venv
- **Platform:** Windows 32-bit (win32)

```bash
# Setup commands
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Development Tools
- **Jupyter Notebooks:** Primary development environment for data exploration
- **Git:** Version control (current branch: main)

## Project Structure

```
thesis-xgboost-explainable-ai/
├── data/
│   ├── raw/              # Original dataset (insurance.csv from Kaggle)
│   └── processed/        # Cleaned and transformed data
├── src/                  # Core Python modules
│   ├── data_preprocessing.py    # InsuranceDataPreprocessor class
│   ├── model_training.py        # XGBoost and baseline models
│   └── explainability.py       # SHAP and LIME implementations
├── notebooks/
│   └── 01_data_exploration.ipynb  # Jupyter analysis notebooks
├── docs/
│   └── phase_progress.md        # Project phase tracking
├── results/
│   ├── models/           # Saved model artifacts
│   ├── plots/            # Generated visualizations
│   └── reports/          # Analysis reports
├── tests/                # Unit tests (pytest structure)
└── venv/                 # Virtual environment (excluded from git)
```

## Tech Stack

### Core ML & Data Science
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: ML preprocessing and baseline models
- **xgboost**: Primary gradient boosting algorithm

### Explainable AI Libraries
- **shap**: SHapley Additive exPlanations for model interpretability
- **lime**: Local Interpretable Model-agnostic Explanations

### Visualization
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualizations
- **plotly**: Interactive plots and dashboards

### Development Environment
- **jupyter**: Notebook environment
- **ipykernel**: Jupyter kernel support
- **tqdm**: Progress bars for long-running operations

## Terminology

### Project-Specific Terms
- **Phase**: Development stages (0-5) as outlined in the thesis timeline
- **Module**: Python files in `src/` directory containing core functionality
- **Explainability**: Refers specifically to SHAP and LIME interpretability methods
- **Baseline Model**: Linear regression reference model for comparison
- **Target Variable**: Medical charges (treatment costs) - the prediction objective

### Dataset Context
- **Insurance Dataset**: Kaggle dataset with 1,338 patient records
- **Features**: age, sex, bmi, children, smoker, region (6 input features)
- **Target**: charges (continuous cost prediction)
- **Split Ratios**: 70% train, 15% validation, 15% test

## Development Phases

### Current Status: Phase 1 - Data Analysis Completed ✅

- [x] **Phase 0:** Environment Setup & GitHub Repository
- [x] **Phase 1:** Data Analysis & Baseline (Week 1) - **EDA COMPLETED**
- [ ] **Phase 2:** XGBoost Implementation (Week 2) 
- [ ] **Phase 3:** Explainable AI Integration (Week 3)
- [ ] **Phase 4:** Dashboard Development (Week 4)
- [ ] **Phase 5:** Documentation & Paper Completion (Week 5-6)

### Phase 1 Key Findings (EDA Results)
Based on comprehensive exploratory data analysis of 1,338 insurance records:

#### Data Quality Assessment
- **Dataset Size**: 1,338 records with 7 features (6 predictors + 1 target)
- **Missing Data**: Minimal - only 3 missing BMI values (0.22%)
- **Data Types**: Mixed dataset with numerical and categorical features
- **Balance**: Well-balanced across demographics (sex ~50/50, regions ~25% each)

#### Target Variable (Charges) Characteristics
- **Distribution**: Highly right-skewed (skewness: 1.516)
- **Range**: $1,121.87 - $63,770.43 (wide variance indicating diverse cost patterns)
- **Central Tendency**: Mean $13,270 vs Median $9,382 (significant difference due to skew)
- **Log Transformation**: Reduces skewness to -0.090 (much more normal)

#### Critical Feature Analysis
**Strongest Predictor - Smoking Status (r=0.787):**
- Smokers pay 280% more than non-smokers ($32,050 vs $8,434 average)
- Only 20.5% of population smokes, but 100% of high-cost cases (top 5%) are smokers
- This represents the dominant cost driver in healthcare charges

**Secondary Predictors:**
- Age (r=0.299): Moderate positive correlation - older patients have higher costs
- BMI (r=0.198): Weak positive correlation, but strong interaction with smoking
- Children (r=0.068): Minimal direct impact on costs
- Sex (r=0.057): Very weak predictor (males ~5% higher costs)
- Region (r=0.006): Negligible regional differences in costs

#### Feature Interactions Discovery
- **BMI × Smoking**: Obese smokers have highest costs ($41,558) vs obese non-smokers ($8,837)
- **Age × Smoking**: Cost increases with age are amplified for smokers
- **High-Risk Profile**: Smokers with BMI >30 represent the most expensive patient segment

#### Outlier Analysis
- 10.4% of cases classified as charge outliers using IQR method
- All top 5% highest cost cases are smokers (67/67 cases)
- Outliers primarily driven by smoking status rather than other factors

#### Implications for XGBoost Modeling
- **Primary Challenge**: Extreme class imbalance in cost distribution
- **Key Advantage**: Clear feature hierarchy with smoking as dominant predictor
- **Feature Engineering**: BMI categories and smoking interactions will be crucial
- **Model Strategy**: Log-transformed target will improve model stability

## Code Conventions

### Python Style
- **Classes**: PascalCase (e.g., `InsuranceDataPreprocessor`)
- **Functions**: snake_case (e.g., `load_insurance_data()`)
- **Constants**: UPPER_SNAKE_CASE
- **Docstrings**: Google-style docstrings for all public methods

### File Organization
- **Imports**: Standard library → Third-party → Local imports
- **Type Hints**: Use modern Python typing (e.g., `pd.DataFrame | None`)
- **Error Handling**: Explicit try-except blocks with meaningful messages

### Data Science Conventions
- **Random State**: Use 42 as default for reproducibility
- **Train/Val/Test**: 70/15/15 split maintained consistently
- **Column Names**: Lowercase with underscores (matching dataset format)

## Common Commands

### Setup and Environment
```bash
# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook
```

### Data Operations
```bash
# Quick data loading (from notebooks)
from src.data_preprocessing import load_insurance_data
df = load_insurance_data()
```

### Model Training (Future Phases)
```bash
# Baseline model
python -c "from src.model_training import train_baseline_model; train_baseline_model()"

# XGBoost training
python -c "from src.model_training import train_xgboost_model; train_xgboost_model()"
```

## Testing and Quality

### Testing Framework
- **pytest**: Primary testing framework (structure set up in `tests/`)
- **Test Coverage**: Aim for >80% coverage on core modules
- **Test Data**: Use subset of insurance dataset for testing

### Code Quality
- **Linting**: Follow PEP 8 standards
- **Type Checking**: Use type hints consistently
- **Documentation**: All public APIs must have docstrings

## Data Guidelines

### Dataset Handling
- **Source Data**: Place `insurance.csv` in `data/raw/`
- **Processing**: Save cleaned data to `data/processed/`
- **Version Control**: Raw data excluded from git (in .gitignore)

### Feature Engineering
- **Categorical Encoding**: Use LabelEncoder for ordinal features
- **Scaling**: StandardScaler for numerical features
- **Target Transform**: Keep charges in original scale for interpretability

## Git Workflow

### Branch Strategy
- **main**: Stable development branch
- **feature/phase-X**: Feature branches for each development phase
- **hotfix/**: Bug fixes requiring immediate attention

### Commit Guidelines
- **Format**: `Phase X: Brief description of changes`
- **Examples**: 
  - `Phase 1: Add data exploration notebook`
  - `Phase 2: Implement XGBoost baseline model`

## Notes for AI Collaboration

### Preferred Patterns
- **Jupyter First**: Start data exploration in notebooks, refactor to modules
- **Incremental Development**: Complete phases sequentially as planned
- **Documentation**: Update this file when adding new conventions or tools

### Common Tasks
1. **Data Analysis**: Use `notebooks/` for exploration, move finalized code to `src/`
2. **Model Development**: Implement in `src/model_training.py` with proper class structure
3. **Visualization**: Save plots to `results/plots/` with descriptive filenames
4. **Testing**: Add tests in `tests/` matching the module structure

### Key Considerations
- **Reproducibility**: Always set random seeds for ML operations
- **Memory Management**: Insurance dataset is small; optimize for code clarity over performance
- **Explainability Focus**: Prioritize interpretable models and clear explanations
- **Academic Standards**: Code quality should meet thesis-level documentation requirements

---

*This file is version-controlled and serves as the single source of truth for AI collaboration on this thesis project. Update as the project evolves through its phases.*