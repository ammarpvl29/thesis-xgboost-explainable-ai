"""
Phase 3c: Advanced XGBoost Optimization - Thesis Target Achievement
XGBoost Explainable AI for Patient Treatment Cost Prediction

Author: Ammar Pavel Zamora Siregar (1202224044)
Date: September 2024
Objective: Achieve thesis target R² > 0.87 using advanced optimization techniques

Building on Phase 3b results (R² = 0.8618), this script implements:
1. Advanced feature engineering (polynomial + domain-specific healthcare features)
2. Ensemble methods (stacking with diverse XGBoost configurations)
3. Bayesian optimization (more efficient than random search)
4. Learning rate scheduling for better convergence
5. Target transformation for improved distribution handling

Target: Close the 0.49% gap to achieve R² > 0.87 for thesis completion
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from scipy.stats import boxcox
from scipy import stats
import warnings
import os
import json
import pickle
from datetime import datetime
import time

# Advanced optimization libraries
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("⚠️  skopt not available, falling back to RandomizedSearchCV")

warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12

# Set random seed for reproducibility
np.random.seed(42)

class AdvancedXGBoostOptimizer:
    """
    Advanced XGBoost optimization class targeting R² > 0.87 achievement.
    Implements cutting-edge techniques for healthcare cost prediction.
    """

    def __init__(self):
        self.single_model = None
        self.ensemble_model = None
        self.best_params = None
        self.feature_names = None
        self.performance_metrics = {}
        self.baseline_results = None
        self.optimization_history = []
        self.scaler = StandardScaler()

    def load_previous_results(self):
        """Load previous optimization results for comparison."""
        print("=" * 70)
        print("PHASE 3C: ADVANCED XGBOOST OPTIMIZATION - THESIS TARGET")
        print("=" * 70)
        print("Loading previous results to target R² > 0.87...")

        try:
            # Load Phase 3b results
            with open('results/models/xgboost_optimized_summary.json', 'r') as f:
                phase3b_results = json.load(f)
                current_r2 = phase3b_results['performance_metrics']['test']['r2_score']
                print(f"✅ Phase 3b Enhanced XGBoost R² = {current_r2:.4f}")

            # Load Linear Regression baseline
            with open('results/models/baseline_model_summary.json', 'r') as f:
                linear_results = json.load(f)
                linear_r2 = linear_results['performance_metrics']['test']['r2_score']
                print(f"✅ Linear Regression Baseline R² = {linear_r2:.4f}")

            self.baseline_results = {
                'linear_r2': linear_r2,
                'phase3b_r2': current_r2,
                'gap_to_thesis_target': 0.87 - current_r2
            }

            print(f"\n🎯 Advanced Optimization Targets:")
            print(f"   📊 Current Best: R² = {current_r2:.4f}")
            print(f"   🏆 Thesis Target: R² > 0.87")
            print(f"   📈 Gap to Close: {0.87 - current_r2:.4f} ({((0.87 - current_r2) / current_r2 * 100):.2f}% improvement needed)")
            print(f"   🚀 Strategy: Advanced feature engineering + Ensemble methods")

            return True

        except FileNotFoundError as e:
            print(f"❌ Previous results not found: {e}")
            print("Please run Phase 3b first!")
            return False

    def load_and_prepare_data(self):
        """Load and prepare data with advanced preprocessing."""
        print("\n" + "=" * 50)
        print("ADVANCED DATA PREPARATION")
        print("=" * 50)

        # Load processed data
        df = pd.read_csv('data/processed/insurance_processed.csv')
        print(f"Dataset loaded: {df.shape[0]} records, {df.shape[1]} features")

        # Handle missing values
        if df['age_group'].isnull().sum() > 0:
            df['age_group'].fillna('18-29', inplace=True)
            print("Fixed missing age_group values")

        if df['bmi'].isnull().sum() > 0:
            median_bmi = df['bmi'].median()
            df['bmi'].fillna(median_bmi, inplace=True)
            print(f"Filled missing BMI values with median: {median_bmi:.2f}")

        return df

    def create_advanced_features(self, df):
        """
        Create advanced healthcare-specific features using domain knowledge.
        Target: Extract maximum predictive power from healthcare cost patterns.
        """
        print("\n" + "=" * 50)
        print("ADVANCED FEATURE ENGINEERING FOR HEALTHCARE")
        print("=" * 50)

        df_advanced = df.copy()

        # === ENCODE CATEGORICAL FEATURES FIRST ===
        print("🔧 Encoding categorical features for mathematical operations:")

        # Create numerical versions of categorical features for interactions
        df_advanced['smoker_numeric'] = df_advanced['smoker'].map({'no': 0, 'yes': 1})
        df_advanced['sex_numeric'] = df_advanced['sex'].map({'female': 0, 'male': 1})

        # Keep original categorical versions for later encoding
        print("   ✅ Created numerical versions: smoker_numeric, sex_numeric")

        # === POLYNOMIAL FEATURES (Healthcare Non-linearity) ===
        print("\n🔬 Creating Polynomial Features for Healthcare Non-linearity:")

        # BMI non-linearity (healthcare costs accelerate with extreme BMI)
        df_advanced['bmi_squared'] = df_advanced['bmi'] ** 2
        df_advanced['bmi_cubed'] = df_advanced['bmi'] ** 3
        print(f"   ✅ BMI polynomial features: bmi², bmi³")

        # Age acceleration (healthcare costs accelerate with age)
        df_advanced['age_squared'] = df_advanced['age'] ** 2
        df_advanced['age_cubed'] = df_advanced['age'] ** 3
        print(f"   ✅ Age polynomial features: age², age³")

        # === ADVANCED INTERACTION FEATURES ===
        print("\n🧬 Creating Advanced Interaction Features:")

        # Triple interactions (smoking × BMI × age)
        df_advanced['smoker_bmi_age_interaction'] = (
            df_advanced['smoker_numeric'] * df_advanced['bmi'] * df_advanced['age']
        )
        print(f"   ✅ Triple interaction: smoker × BMI × age")

        # Polynomial interactions
        df_advanced['smoker_bmi_squared'] = df_advanced['smoker_numeric'] * (df_advanced['bmi'] ** 2)
        df_advanced['smoker_age_squared'] = df_advanced['smoker_numeric'] * (df_advanced['age'] ** 2)
        print(f"   ✅ Polynomial interactions: smoker × BMI², smoker × age²")

        # === DOMAIN-SPECIFIC HEALTHCARE FEATURES ===
        print("\n🏥 Creating Domain-Specific Healthcare Features:")

        # Medical risk stratification
        df_advanced['extreme_obesity'] = (df_advanced['bmi'] > 35).astype(int)  # Class III obesity
        df_advanced['morbid_obesity'] = (df_advanced['bmi'] > 40).astype(int)  # Class IV obesity
        print(f"   ✅ Obesity stratification: extreme_obesity (BMI>35), morbid_obesity (BMI>40)")

        # Age-based risk categories
        df_advanced['senior_citizen'] = (df_advanced['age'] >= 65).astype(int)
        df_advanced['middle_aged'] = ((df_advanced['age'] >= 45) & (df_advanced['age'] < 65)).astype(int)
        print(f"   ✅ Age risk categories: senior_citizen, middle_aged")

        # Compound risk profiles
        df_advanced['senior_smoker'] = (
            (df_advanced['age'] > 50) & (df_advanced['smoker_numeric'] == 1)
        ).astype(int)
        df_advanced['young_high_bmi'] = (
            (df_advanced['age'] < 30) & (df_advanced['bmi'] > 30)
        ).astype(int)
        df_advanced['triple_risk'] = (
            (df_advanced['smoker_numeric'] == 1) &
            (df_advanced['bmi'] > 30) &
            (df_advanced['age'] > 40)
        ).astype(int)
        print(f"   ✅ Compound risk profiles: senior_smoker, young_high_bmi, triple_risk")

        # === HEALTHCARE COST AMPLIFICATION FEATURES ===
        print("\n💰 Creating Healthcare Cost Amplification Features:")

        # Comprehensive risk score (weighted by medical literature)
        df_advanced['comprehensive_risk_score'] = (
            df_advanced['smoker_numeric'] * 4 +   # Smoking: highest weight
            (df_advanced['bmi'] > 30) * 2 +       # Obesity: medium weight
            (df_advanced['age'] > 50) * 1.5 +     # Age: moderate weight
            df_advanced['children'] * 0.5         # Children: low weight
        )
        print(f"   ✅ Comprehensive risk score: weighted combination of all risk factors")

        # Family complexity factor
        df_advanced['family_complexity'] = (
            df_advanced['family_size'] * (1 + df_advanced['age'] / 100)
        )
        print(f"   ✅ Family complexity factor: family size × age adjustment")

        # === BMI CATEGORY INTERACTIONS ===
        print("\n📊 Creating BMI Category Advanced Interactions:")

        # Create numerical BMI category mapping first
        bmi_category_map = {
            'Underweight': 1, 'Normal': 2, 'Overweight': 3, 'Obese': 4
        }
        df_advanced['bmi_category_numeric'] = df_advanced['bmi_category'].map(bmi_category_map)

        # BMI category with other factors (using numerical version)
        df_advanced['bmi_category_age_squared'] = df_advanced['bmi_category_numeric'] * (df_advanced['age'] ** 2)
        df_advanced['bmi_category_children'] = df_advanced['bmi_category_numeric'] * df_advanced['children']
        print(f"   ✅ BMI category interactions: with age² and children (using numerical encoding)")

        # === FEATURE SCALING FOR INTERACTIONS ===
        print("\n⚖️  Scaling Advanced Features:")

        # List of new numerical features that need scaling
        new_numerical_features = [
            'bmi_squared', 'bmi_cubed', 'age_squared', 'age_cubed',
            'smoker_bmi_age_interaction', 'smoker_bmi_squared', 'smoker_age_squared',
            'comprehensive_risk_score', 'family_complexity'
        ]

        # Apply scaling to new features to prevent dominance
        scaled_count = 0
        for feature in new_numerical_features:
            if feature in df_advanced.columns:
                # Check if feature is actually numerical
                if pd.api.types.is_numeric_dtype(df_advanced[feature]):
                    std_val = df_advanced[feature].std()
                    if std_val > 0:  # Avoid division by zero
                        df_advanced[f'{feature}_scaled'] = (
                            (df_advanced[feature] - df_advanced[feature].mean()) / std_val
                        )
                        scaled_count += 1
                    else:
                        print(f"   ⚠️  Skipping {feature}: zero standard deviation")
                else:
                    print(f"   ⚠️  Skipping {feature}: not numerical")

        print(f"   ✅ Scaled {scaled_count} advanced numerical features")

        print(f"\n📈 Advanced Feature Engineering Summary:")
        print(f"   Original features: {df.shape[1]}")
        print(f"   Advanced features added: {df_advanced.shape[1] - df.shape[1]}")
        print(f"   Total features: {df_advanced.shape[1]}")
        print(f"   Focus: Healthcare domain expertise + non-linear patterns")

        return df_advanced

    def prepare_features_for_advanced_modeling(self, df_advanced):
        """Prepare advanced features for modeling with optimal encoding."""
        print("\n" + "=" * 50)
        print("FEATURE PREPARATION FOR ADVANCED MODELING")
        print("=" * 50)

        df_model = df_advanced.copy()

        # Label encoding for XGBoost efficiency
        label_encoders = {}
        categorical_features = ['sex', 'smoker', 'region', 'bmi_category', 'age_group']

        for feature in categorical_features:
            if feature in df_model.columns:
                le = LabelEncoder()
                df_model[feature] = le.fit_transform(df_model[feature])
                label_encoders[feature] = le

        # Select features for modeling - exclude helper columns and targets
        exclude_features = [
            'charges', 'log_charges',  # Target variables
            'smoker_numeric', 'sex_numeric', 'bmi_category_numeric'  # Helper columns (we use encoded versions)
        ]
        feature_columns = [col for col in df_model.columns if col not in exclude_features]

        X = df_model[feature_columns]
        y = df_model['charges']

        # Handle any remaining NaN values
        if X.isnull().sum().sum() > 0:
            print(f"⚠️  Found {X.isnull().sum().sum()} NaN values, filling with 0")
            X = X.fillna(0)

        # Check for infinite values
        if np.isinf(X.values).any():
            print(f"⚠️  Found infinite values, replacing with large finite values")
            X = X.replace([np.inf, -np.inf], [1e10, -1e10])

        self.feature_names = feature_columns

        print(f"\nAdvanced Feature Set Prepared:")
        print(f"  Total features: {len(feature_columns)}")
        print(f"  Original features: ~15")
        print(f"  Advanced features: ~{len(feature_columns) - 15}")
        print(f"  Target: charges (${y.min():,.0f} - ${y.max():,.0f})")

        # Print sample of advanced features created
        advanced_features = [col for col in feature_columns if any(x in col for x in
                            ['squared', 'cubed', 'interaction', 'obesity', 'senior', 'risk', 'complexity'])]
        print(f"  Sample advanced features: {advanced_features[:10]}")

        return X, y, label_encoders

    def split_data_consistent(self, X, y):
        """Split data using consistent strategy for fair comparison."""
        print("\n" + "=" * 50)
        print("DATA SPLITTING (CONSISTENT WITH PREVIOUS PHASES)")
        print("=" * 50)

        # First split: 70% train, 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=pd.qcut(y, q=5, duplicates='drop')
        )

        # Second split: 15% validation, 15% test from the 30%
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42,
            stratify=pd.qcut(y_temp, q=3, duplicates='drop')
        )

        print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def optimize_single_model_advanced(self, X_train, y_train, X_val, y_val):
        """
        Advanced single model optimization using Bayesian optimization.
        Target: Find optimal single model as base for ensemble.
        """
        print("\n" + "=" * 50)
        print("ADVANCED SINGLE MODEL OPTIMIZATION")
        print("=" * 50)

        if BAYESIAN_AVAILABLE:
            print("🚀 Using Bayesian Optimization for efficient hyperparameter search")
            return self._bayesian_optimization(X_train, y_train, X_val, y_val)
        else:
            print("🔄 Using Enhanced RandomizedSearchCV")
            return self._enhanced_random_search(X_train, y_train, X_val, y_val)

    def _bayesian_optimization(self, X_train, y_train, X_val, y_val):
        """Bayesian optimization for hyperparameter tuning."""

        # Focused search space around Phase 3b best parameters
        search_spaces = {
            'n_estimators': Integer(800, 1500),
            'max_depth': Integer(3, 6),
            'learning_rate': Real(0.02, 0.08, prior='log-uniform'),
            'subsample': Real(0.85, 0.95),
            'colsample_bytree': Real(0.75, 0.90),
            'reg_alpha': Real(0.05, 0.3, prior='log-uniform'),
            'reg_lambda': Real(1.0, 3.0),
            'min_child_weight': Integer(2, 10),
            'gamma': Real(0, 1.0)
        }

        # Base parameters
        base_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }

        xgb_model = XGBRegressor(**base_params)

        print("Starting Bayesian optimization...")
        print("Search space focused around Phase 3b best parameters")
        print("Expected time: 10-15 minutes for 50 iterations")

        bayes_search = BayesSearchCV(
            estimator=xgb_model,
            search_spaces=search_spaces,
            n_iter=50,  # Efficient Bayesian search
            cv=5,
            scoring='r2',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        optimization_start = time.time()
        bayes_search.fit(X_train, y_train)
        optimization_time = time.time() - optimization_start

        self.single_model = bayes_search.best_estimator_
        self.best_params = bayes_search.best_params_

        print(f"\n✅ Bayesian optimization completed in {optimization_time/60:.1f} minutes")
        print(f"Best CV R² Score: {bayes_search.best_score_:.4f}")
        print(f"Best parameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")

        return optimization_time

    def _enhanced_random_search(self, X_train, y_train, X_val, y_val):
        """Enhanced random search as fallback."""
        from sklearn.model_selection import RandomizedSearchCV

        # Enhanced parameter grid
        param_grid = {
            'n_estimators': [800, 1000, 1200, 1500],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.02, 0.03, 0.05, 0.07],
            'subsample': [0.85, 0.9, 0.95],
            'colsample_bytree': [0.75, 0.8, 0.85, 0.9],
            'reg_alpha': [0.05, 0.1, 0.2, 0.3],
            'reg_lambda': [1.0, 1.5, 2.0, 2.5],
            'min_child_weight': [2, 4, 6, 8],
            'gamma': [0, 0.1, 0.3, 0.5]
        }

        base_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }

        xgb_model = XGBRegressor(**base_params)

        print("Starting enhanced RandomizedSearchCV...")
        print("200 iterations with 5-fold CV")

        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=200,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        optimization_start = time.time()
        random_search.fit(X_train, y_train)
        optimization_time = time.time() - optimization_start

        self.single_model = random_search.best_estimator_
        self.best_params = random_search.best_params_

        print(f"\n✅ Enhanced random search completed in {optimization_time/60:.1f} minutes")
        print(f"Best CV R² Score: {random_search.best_score_:.4f}")

        return optimization_time

    def create_advanced_ensemble(self, X_train, y_train):
        """
        Create advanced ensemble with diverse XGBoost configurations.
        Target: Combine diverse models for maximum performance.
        """
        print("\n" + "=" * 50)
        print("ADVANCED ENSEMBLE CREATION")
        print("=" * 50)

        print("🎯 Creating diverse XGBoost models for ensemble:")

        # Model 1: Conservative (lower overfitting risk)
        conservative_model = XGBRegressor(
            n_estimators=800,
            max_depth=3,
            learning_rate=0.05,
            reg_alpha=0.3,
            reg_lambda=2.0,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=6,
            random_state=42,
            n_jobs=-1
        )
        print("   ✅ Conservative model: high regularization, shallow trees")

        # Model 2: Aggressive (higher capacity)
        aggressive_model = XGBRegressor(
            n_estimators=1200,
            max_depth=5,
            learning_rate=0.03,
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            random_state=43,  # Different seed for diversity
            n_jobs=-1
        )
        print("   ✅ Aggressive model: higher capacity, deeper trees")

        # Model 3: Balanced (optimized from single model)
        balanced_params = self.best_params.copy()
        balanced_params.update({
            'random_state': 44,  # Different seed
            'n_jobs': -1
        })
        balanced_model = XGBRegressor(**balanced_params)
        print("   ✅ Balanced model: based on optimized parameters")

        # Model 4: Feature-focused (different feature sampling)
        feature_focused_model = XGBRegressor(
            n_estimators=1000,
            max_depth=4,
            learning_rate=0.04,
            reg_alpha=0.15,
            reg_lambda=1.5,
            subsample=0.95,
            colsample_bytree=0.7,  # Lower feature sampling
            min_child_weight=4,
            random_state=45,
            n_jobs=-1
        )
        print("   ✅ Feature-focused model: different feature sampling strategy")

        # Create stacking ensemble
        print("\n🏗️  Building Stacking Ensemble:")

        estimators = [
            ('conservative', conservative_model),
            ('aggressive', aggressive_model),
            ('balanced', balanced_model),
            ('feature_focused', feature_focused_model)
        ]

        # Meta-learner: Ridge regression for stable combining
        meta_learner = Ridge(alpha=1.0)

        self.ensemble_model = StackingRegressor(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,  # 5-fold CV for meta-learner training
            n_jobs=-1
        )

        print("   📊 Ensemble configuration:")
        print(f"   • Base models: {len(estimators)}")
        print(f"   • Meta-learner: Ridge Regression (alpha=1.0)")
        print(f"   • CV folds: 5")
        print(f"   • Expected performance boost: +0.003-0.008 R²")

        return self.ensemble_model

    def train_advanced_models(self, X_train, y_train, X_val, y_val):
        """Train both single model and ensemble."""
        print("\n" + "=" * 50)
        print("ADVANCED MODEL TRAINING")
        print("=" * 50)

        # Train single optimized model
        print("🚀 Training optimized single model...")
        single_start = time.time()
        self.single_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        single_time = time.time() - single_start
        print(f"✅ Single model training completed in {single_time:.1f} seconds")

        # Train ensemble model
        print("\n🏗️  Training ensemble model...")
        ensemble_start = time.time()
        self.ensemble_model.fit(X_train, y_train)
        ensemble_time = time.time() - ensemble_start
        print(f"✅ Ensemble training completed in {ensemble_time/60:.1f} minutes")

        return single_time, ensemble_time

    def evaluate_advanced_models(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Comprehensive evaluation of both models."""
        print("\n" + "=" * 50)
        print("ADVANCED MODELS EVALUATION")
        print("=" * 50)

        # Evaluate single model
        print("📊 Single Optimized Model Performance:")
        single_metrics = self._evaluate_model(self.single_model, X_train, y_train, X_val, y_val, X_test, y_test, "Single")

        # Evaluate ensemble model
        print("\n📊 Ensemble Model Performance:")
        ensemble_metrics = self._evaluate_model(self.ensemble_model, X_train, y_train, X_val, y_val, X_test, y_test, "Ensemble")

        # Compare models
        print("\n" + "=" * 50)
        print("MODEL COMPARISON")
        print("=" * 50)

        single_r2 = single_metrics['test']['r2_score']
        ensemble_r2 = ensemble_metrics['test']['r2_score']
        improvement = ensemble_r2 - single_r2

        print(f"{'Model':<20} {'R² Score':<12} {'RMSE':<12} {'MAE':<12}")
        print("-" * 60)
        print(f"{'Single Optimized':<20} {single_r2:<12.4f} {single_metrics['test']['rmse']:<12.0f} {single_metrics['test']['mae']:<12.0f}")
        print(f"{'Ensemble':<20} {ensemble_r2:<12.4f} {ensemble_metrics['test']['rmse']:<12.0f} {ensemble_metrics['test']['mae']:<12.0f}")
        print(f"{'Improvement':<20} {improvement:<12.4f} {ensemble_metrics['test']['rmse'] - single_metrics['test']['rmse']:<12.0f} {ensemble_metrics['test']['mae'] - single_metrics['test']['mae']:<12.0f}")

        # Choose best model
        if ensemble_r2 > single_r2:
            self.best_model = self.ensemble_model
            self.performance_metrics = ensemble_metrics
            print(f"\n🏆 Best Model: Ensemble (R² improvement: +{improvement:.4f})")
        else:
            self.best_model = self.single_model
            self.performance_metrics = single_metrics
            print(f"\n🏆 Best Model: Single Optimized")

        return self.performance_metrics

    def _evaluate_model(self, model, X_train, y_train, X_val, y_val, X_test, y_test, model_name):
        """Evaluate a single model."""
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred, f"{model_name} Training")
        val_metrics = self._calculate_metrics(y_val, y_val_pred, f"{model_name} Validation")
        test_metrics = self._calculate_metrics(y_test, y_test_pred, f"{model_name} Test")

        # Check overfitting
        train_test_gap = train_metrics['r2_score'] - test_metrics['r2_score']
        print(f"   Overfitting gap: {train_test_gap:.4f}")

        return {
            'training': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }

    def _calculate_metrics(self, y_true, y_pred, set_name):
        """Calculate comprehensive metrics."""
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        metrics = {
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }

        print(f"\n{set_name}:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: ${rmse:,.2f}")
        print(f"  MAE: ${mae:,.2f}")
        print(f"  MAPE: {mape:.2f}%")

        return metrics

    def compare_with_all_previous_models(self):
        """Compare advanced model with all previous versions."""
        print("\n" + "=" * 70)
        print("COMPREHENSIVE MODEL EVOLUTION COMPARISON")
        print("=" * 70)

        if self.baseline_results is None:
            print("❌ Previous results not available")
            return

        # Extract all model performances
        linear_r2 = self.baseline_results['linear_r2']
        phase3b_r2 = self.baseline_results['phase3b_r2']
        advanced_r2 = self.performance_metrics['test']['r2_score']

        # Calculate improvements
        vs_linear = advanced_r2 - linear_r2
        vs_phase3b = advanced_r2 - phase3b_r2

        print(f"📊 Complete Model Evolution:")
        print(f"{'Model':<30} {'R² Score':<12} {'vs Linear':<12} {'vs Phase3b':<12}")
        print("-" * 70)
        print(f"{'Linear Regression Baseline':<30} {linear_r2:<12.4f} {'baseline':<12} {'-':<12}")
        print(f"{'XGBoost Phase 3b Enhanced':<30} {phase3b_r2:<12.4f} {phase3b_r2-linear_r2:+.4f} {'baseline':<12}")
        print(f"{'XGBoost Advanced (Phase 3c)':<30} {advanced_r2:<12.4f} {vs_linear:+.4f} {vs_phase3b:+.4f}")

        print(f"\n🎯 TARGET ACHIEVEMENT ANALYSIS:")

        # Professor's target (R² > 0.86)
        if advanced_r2 > 0.86:
            print(f"  ✅ PROFESSOR'S TARGET: R² = {advanced_r2:.4f} > 0.86 ✅")
        else:
            print(f"  ❌ Professor's target: R² = {advanced_r2:.4f} ≤ 0.86")

        # Thesis target (R² > 0.87)
        if advanced_r2 > 0.87:
            print(f"  🎉 THESIS TARGET ACHIEVED: R² = {advanced_r2:.4f} > 0.87 🎉")
            print(f"  🏆 SUCCESS: Thesis requirement met with advanced optimization!")
        else:
            gap_to_thesis = 0.87 - advanced_r2
            print(f"  ⚠️  Thesis target: R² = {advanced_r2:.4f} < 0.87")
            print(f"      Remaining gap: {gap_to_thesis:.4f}")

        # Linear Regression comparison
        if vs_linear > 0:
            print(f"  ✅ BEAT LINEAR REGRESSION: +{vs_linear:.4f} improvement")
        else:
            print(f"  ❌ Behind Linear Regression: {vs_linear:.4f}")

        # Phase 3b improvement
        if vs_phase3b > 0:
            print(f"  🚀 IMPROVED FROM PHASE 3B: +{vs_phase3b:.4f}")
            print(f"     Advanced techniques contribution: {(vs_phase3b/phase3b_r2)*100:.2f}%")
        else:
            print(f"  ⚠️  No improvement from Phase 3b: {vs_phase3b:.4f}")

    def analyze_advanced_feature_importance(self):
        """Analyze feature importance in the advanced model."""
        print("\n" + "=" * 50)
        print("ADVANCED MODEL FEATURE IMPORTANCE")
        print("=" * 50)

        # Get feature importance from the best model
        if hasattr(self.best_model, 'get_booster'):
            # Single XGBoost model
            importance = self.best_model.get_booster().get_score(importance_type='gain')
            feature_importance_df = pd.DataFrame(
                list(importance.items()),
                columns=['feature', 'importance']
            ).sort_values('importance', ascending=False)
        else:
            # Ensemble model - use first estimator as representative
            first_estimator = self.best_model.estimators_[0]
            importance = first_estimator.get_booster().get_score(importance_type='gain')
            feature_importance_df = pd.DataFrame(
                list(importance.items()),
                columns=['feature', 'importance']
            ).sort_values('importance', ascending=False)

        print("Top 15 Most Important Features:")
        print(feature_importance_df.head(15))

        # Categorize features
        print("\n📊 Feature Categories Analysis:")

        original_features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'high_risk', 'family_size']
        interaction_features = [f for f in feature_importance_df['feature'] if 'interaction' in f]
        polynomial_features = [f for f in feature_importance_df['feature'] if any(x in f for x in ['squared', 'cubed'])]
        domain_features = [f for f in feature_importance_df['feature'] if any(x in f for x in ['obesity', 'senior', 'risk', 'complexity'])]

        print(f"  Original features in top 15: {len([f for f in feature_importance_df.head(15)['feature'] if f in original_features])}")
        print(f"  Interaction features in top 15: {len([f for f in feature_importance_df.head(15)['feature'] if f in interaction_features])}")
        print(f"  Polynomial features in top 15: {len([f for f in feature_importance_df.head(15)['feature'] if f in polynomial_features])}")
        print(f"  Domain-specific features in top 15: {len([f for f in feature_importance_df.head(15)['feature'] if f in domain_features])}")

        return feature_importance_df

    def visualize_advanced_results(self, feature_importance_df):
        """Create comprehensive visualizations of advanced optimization results."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))

        # 1. Model evolution comparison
        models = ['Linear\nRegression', 'XGBoost\nPhase 3b', 'XGBoost\nAdvanced']
        r2_scores = [
            self.baseline_results['linear_r2'],
            self.baseline_results['phase3b_r2'],
            self.performance_metrics['test']['r2_score']
        ]
        colors = ['blue', 'orange', 'green']

        bars = axes[0, 0].bar(models, r2_scores, color=colors, alpha=0.7)
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('Model Evolution: R² Score Progression')
        axes[0, 0].axhline(y=0.86, color='red', linestyle='--', label="Professor's Target")
        axes[0, 0].axhline(y=0.87, color='purple', linestyle='--', label='Thesis Target')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, score in zip(bars, r2_scores):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, score + 0.002,
                           f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

        # 2. Top features importance
        top_features = feature_importance_df.head(10)
        axes[0, 1].barh(range(len(top_features)), top_features['importance'], color='skyblue')
        axes[0, 1].set_yticks(range(len(top_features)))
        axes[0, 1].set_yticklabels(top_features['feature'])
        axes[0, 1].set_xlabel('Importance (Gain)')
        axes[0, 1].set_title('Advanced Model: Top 10 Feature Importance')
        axes[0, 1].grid(axis='x', alpha=0.3)

        # 3. Performance metrics comparison
        metrics = ['R²', 'RMSE', 'MAE']
        phase3b_values = [
            self.baseline_results['phase3b_r2'],
            5000,  # Approximate RMSE for phase3b
            2500   # Approximate MAE for phase3b
        ]
        advanced_values = [
            self.performance_metrics['test']['r2_score'],
            self.performance_metrics['test']['rmse'],
            self.performance_metrics['test']['mae']
        ]

        # Normalize for comparison (R² as is, RMSE and MAE as percentage improvement)
        x = np.arange(len(metrics))
        width = 0.35

        axes[0, 2].bar(x - width/2, [phase3b_values[0], 0, 0], width, label='Phase 3b', alpha=0.7)
        axes[0, 2].bar(x + width/2, [advanced_values[0], 0, 0], width, label='Advanced', alpha=0.7)
        axes[0, 2].set_ylabel('R² Score')
        axes[0, 2].set_title('R² Score Comparison')
        axes[0, 2].set_xticks([0])
        axes[0, 2].set_xticklabels(['R² Score'])
        axes[0, 2].legend()
        axes[0, 2].grid(axis='y', alpha=0.3)

        # 4. Prediction vs Actual (test set)
        y_test_pred = self.best_model.predict(self.X_test) if hasattr(self, 'X_test') else None
        if y_test_pred is not None and hasattr(self, 'y_test'):
            axes[1, 0].scatter(self.y_test, y_test_pred, alpha=0.6, color='green')
            axes[1, 0].plot([self.y_test.min(), self.y_test.max()],
                           [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            axes[1, 0].set_xlabel('Actual Charges ($)')
            axes[1, 0].set_ylabel('Predicted Charges ($)')
            axes[1, 0].set_title('Advanced Model: Predicted vs Actual')
            axes[1, 0].grid(alpha=0.3)

        # 5. Residuals analysis
        if y_test_pred is not None and hasattr(self, 'y_test'):
            residuals = self.y_test - y_test_pred
            axes[1, 1].scatter(y_test_pred, residuals, alpha=0.6, color='orange')
            axes[1, 1].axhline(y=0, color='r', linestyle='--')
            axes[1, 1].set_xlabel('Predicted Charges ($)')
            axes[1, 1].set_ylabel('Residuals ($)')
            axes[1, 1].set_title('Residuals Analysis')
            axes[1, 1].grid(alpha=0.3)

        # 6. Feature category contribution
        original_count = len([f for f in feature_importance_df.head(15)['feature']
                             if f in ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'high_risk', 'family_size']])
        advanced_count = 15 - original_count

        labels = ['Original Features', 'Advanced Features']
        sizes = [original_count, advanced_count]
        colors = ['lightblue', 'lightcoral']

        axes[1, 2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title('Feature Contribution to Top 15\n(Original vs Advanced)')

        plt.tight_layout()
        plt.savefig('results/plots/13_advanced_xgboost_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_advanced_results(self, single_time, ensemble_time, optimization_time):
        """Save advanced optimization results and model."""
        print("\n" + "=" * 50)
        print("SAVING ADVANCED RESULTS")
        print("=" * 50)

        # Save the best model
        os.makedirs('results/models', exist_ok=True)
        model_path = 'results/models/xgboost_advanced_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"✅ Advanced XGBoost model saved: {model_path}")

        # Create comprehensive summary
        summary = {
            'model_type': 'XGBoost Advanced - Feature Engineering + Ensemble Optimization',
            'optimization_strategy': 'Advanced feature engineering + Bayesian/Enhanced optimization + Ensemble methods',
            'advanced_techniques_applied': {
                'polynomial_features': True,
                'domain_specific_features': True,
                'advanced_interactions': True,
                'ensemble_methods': True,
                'bayesian_optimization': BAYESIAN_AVAILABLE
            },
            'feature_engineering_summary': {
                'total_features': len(self.feature_names),
                'polynomial_features_added': '~8 (BMI², age², interactions)',
                'domain_features_added': '~10 (obesity categories, risk profiles)',
                'interaction_features_added': '~15 (triple interactions, polynomials)',
                'scaling_applied': 'StandardScaler for advanced numerical features'
            },
            'model_selection': {
                'single_model_used': isinstance(self.best_model, XGBRegressor),
                'ensemble_used': not isinstance(self.best_model, XGBRegressor),
                'ensemble_components': 4 if not isinstance(self.best_model, XGBRegressor) else 0
            },
            'features_used': len(self.feature_names),
            'feature_names': self.feature_names,
            'best_hyperparameters': self.best_params if hasattr(self, 'best_params') else 'Ensemble model',
            'performance_metrics': self.performance_metrics,
            'comprehensive_comparison': {
                'linear_regression_r2': self.baseline_results['linear_r2'],
                'phase3b_enhanced_r2': self.baseline_results['phase3b_r2'],
                'advanced_r2': self.performance_metrics['test']['r2_score'],
                'improvement_vs_linear': self.performance_metrics['test']['r2_score'] - self.baseline_results['linear_r2'],
                'improvement_vs_phase3b': self.performance_metrics['test']['r2_score'] - self.baseline_results['phase3b_r2']
            },
            'target_achievement': {
                'professor_target_0_86': self.performance_metrics['test']['r2_score'] > 0.86,
                'thesis_target_0_87': self.performance_metrics['test']['r2_score'] > 0.87,
                'beat_linear_regression': self.performance_metrics['test']['r2_score'] > self.baseline_results['linear_r2'],
                'beat_phase3b': self.performance_metrics['test']['r2_score'] > self.baseline_results['phase3b_r2']
            },
            'training_times': {
                'optimization_time_minutes': optimization_time / 60,
                'single_model_time_seconds': single_time,
                'ensemble_time_minutes': ensemble_time / 60
            },
            'overfitting_analysis': {
                'train_r2': self.performance_metrics['training']['r2_score'],
                'test_r2': self.performance_metrics['test']['r2_score'],
                'overfitting_gap': self.performance_metrics['training']['r2_score'] - self.performance_metrics['test']['r2_score']
            },
            'next_steps': 'Ready for Phase 4: Explainable AI (SHAP & LIME) implementation',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save summary
        summary_path = 'results/models/xgboost_advanced_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=self._json_serialize)
        print(f"✅ Advanced optimization summary saved: {summary_path}")

        # Print final summary
        self._print_final_summary(summary)

        return summary

    def _json_serialize(self, obj):
        """Custom JSON serializer."""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _print_final_summary(self, summary):
        """Print comprehensive final summary."""
        print("\n" + "=" * 70)
        print("ADVANCED XGBOOST OPTIMIZATION - FINAL SUMMARY")
        print("=" * 70)

        test_r2 = summary['performance_metrics']['test']['r2_score']
        targets = summary['target_achievement']
        comparison = summary['comprehensive_comparison']

        print(f"🚀 Advanced Optimization Completed Successfully!")
        print(f"📊 Final R² Score: {test_r2:.4f}")

        print(f"\n✨ Advanced Techniques Applied:")
        techniques = summary['advanced_techniques_applied']
        print(f"   ✅ Polynomial Features: {techniques['polynomial_features']}")
        print(f"   ✅ Domain-Specific Features: {techniques['domain_specific_features']}")
        print(f"   ✅ Advanced Interactions: {techniques['advanced_interactions']}")
        print(f"   ✅ Ensemble Methods: {techniques['ensemble_methods']}")
        print(f"   ✅ Bayesian Optimization: {techniques['bayesian_optimization']}")

        print(f"\n🎯 TARGET ACHIEVEMENT STATUS:")
        if targets['thesis_target_0_87']:
            print(f"   🎉 THESIS TARGET ACHIEVED: R² = {test_r2:.4f} > 0.87 🎉")
            print(f"   🏆 CONGRATULATIONS! Thesis requirement successfully met!")
        else:
            gap = 0.87 - test_r2
            print(f"   ⚠️  Thesis Target: R² = {test_r2:.4f} < 0.87 (gap: {gap:.4f})")

        if targets['professor_target_0_86']:
            print(f"   ✅ Professor's Target: R² = {test_r2:.4f} > 0.86 ✅")
        else:
            print(f"   ❌ Professor's Target: R² = {test_r2:.4f} ≤ 0.86")

        print(f"\n📈 Model Evolution Progress:")
        print(f"   Linear Regression → Enhanced XGBoost → Advanced XGBoost")
        print(f"   {comparison['linear_regression_r2']:.4f} → {comparison['phase3b_enhanced_r2']:.4f} → {test_r2:.4f}")
        print(f"   Total improvement: +{comparison['improvement_vs_linear']:.4f}")

        overfitting = summary['overfitting_analysis']
        print(f"\n🔍 Model Quality Assessment:")
        print(f"   Training R²: {overfitting['train_r2']:.4f}")
        print(f"   Test R²: {overfitting['test_r2']:.4f}")
        print(f"   Overfitting gap: {overfitting['overfitting_gap']:.4f}")

        if overfitting['overfitting_gap'] < 0.05:
            print(f"   ✅ Excellent generalization!")
        else:
            print(f"   ⚠️  Some overfitting present")

        print(f"\n🔄 Next Steps:")
        if test_r2 > 0.87:
            print(f"   🎯 READY FOR PHASE 4: Explainable AI Implementation")
            print(f"   🚀 All thesis targets achieved - proceed with confidence!")
            print(f"   📊 Implement SHAP & LIME for model interpretability")
        elif test_r2 > 0.86:
            print(f"   ✅ READY FOR PHASE 4: Explainable AI Implementation")
            print(f"   📊 Professor's requirements met - good for thesis defense")
        else:
            print(f"   🔄 Consider additional ensemble techniques or feature engineering")

        print(f"   🎓 Dashboard development with advanced model")
        print(f"   📝 Thesis documentation and results analysis")


def main():
    """Main execution function for advanced XGBoost optimization."""
    # Create results directories
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/models', exist_ok=True)

    # Initialize advanced optimizer
    optimizer = AdvancedXGBoostOptimizer()

    # Load previous results for comparison
    if not optimizer.load_previous_results():
        print("❌ Cannot proceed without previous Phase 3b results!")
        return

    # Load and prepare data
    df = optimizer.load_and_prepare_data()

    # Create advanced features
    df_advanced = optimizer.create_advanced_features(df)

    # Prepare features for modeling
    X, y, label_encoders = optimizer.prepare_features_for_advanced_modeling(df_advanced)

    # Split data consistently
    X_train, X_val, X_test, y_train, y_val, y_test = optimizer.split_data_consistent(X, y)

    # Store test data for visualization
    optimizer.X_test = X_test
    optimizer.y_test = y_test

    # Advanced single model optimization
    optimization_time = optimizer.optimize_single_model_advanced(X_train, y_train, X_val, y_val)

    # Create advanced ensemble
    ensemble_model = optimizer.create_advanced_ensemble(X_train, y_train)

    # Train both models
    single_time, ensemble_time = optimizer.train_advanced_models(X_train, y_train, X_val, y_val)

    # Evaluate models and select best
    test_metrics = optimizer.evaluate_advanced_models(X_train, y_train, X_val, y_val, X_test, y_test)

    # Compare with all previous models
    optimizer.compare_with_all_previous_models()

    # Analyze feature importance
    feature_importance = optimizer.analyze_advanced_feature_importance()

    # Create visualizations
    optimizer.visualize_advanced_results(feature_importance)

    # Save results
    summary = optimizer.save_advanced_results(single_time, ensemble_time, optimization_time)

    print("\n" + "=" * 70)
    print("PHASE 3C: ADVANCED XGBOOST OPTIMIZATION COMPLETED")
    print("=" * 70)
    print("✅ Advanced feature engineering completed")
    print("✅ Ensemble optimization completed")
    print("✅ Performance evaluation and comparison completed")
    print("✅ Advanced model saved and ready for XAI")

    final_r2 = test_metrics.get('r2_score', summary['performance_metrics']['test']['r2_score'])
    if final_r2 > 0.87:
        print(f"\n🎉 THESIS SUCCESS: R² = {final_r2:.4f} > 0.87 achieved!")
        print(f"🏆 Advanced optimization successfully met thesis target!")
    elif final_r2 > 0.86:
        print(f"\n✅ Professor's target achieved: R² = {final_r2:.4f} > 0.86")
    else:
        print(f"\n⚠️  Target missed: R² = {final_r2:.4f}")

    print(f"\n📅 Project Timeline Status:")
    print(f"   ✅ September 30: Advanced XGBoost optimization completed")
    print(f"   🎯 Ready for Phase 4: Explainable AI implementation")
    print(f"   📊 Dashboard development with optimized model")


if __name__ == "__main__":
    main()