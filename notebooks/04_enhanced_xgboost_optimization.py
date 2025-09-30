"""
Phase 4: Enhanced XGBoost Hyperparameter Optimization
XGBoost Explainable AI for Patient Treatment Cost Prediction

Author: Ammar Pavel Zamora Siregar (1202224044)
Date: September 2024
Objective: Achieve R² > 0.87 through systematic hyperparameter optimization

Current Situation Analysis:
- Enhanced Linear Regression: R² = 0.8566 (strong baseline)
- Enhanced XGBoost Baseline: R² = 0.8014 (underperformed due to overfitting)
- Overfitting Gap: 0.1975 (Training R² = 0.9989 vs Test R² = 0.8014)
- Feature Quality: Excellent (high_risk, smoker_bmi_interaction dominate)

Optimization Strategy:
1. PRIORITY: Reduce overfitting through aggressive regularization
2. TARGET: Achieve R² > 0.87 (thesis requirement)
3. BENCHMARK: Beat Enhanced Linear Regression (R² > 0.8566)
4. METHOD: Systematic hyperparameter search focused on generalization

Expected Outcome: R² > 0.87 with proper overfitting control
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import json
import pickle
from datetime import datetime
import time
from scipy import stats

# Advanced optimization libraries
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    BAYESIAN_AVAILABLE = True
    print("✅ Bayesian optimization available")
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("⚠️  Bayesian optimization not available, using RandomizedSearchCV")

warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 12

# Set random seed for reproducibility
np.random.seed(42)

class EnhancedXGBoostOptimizer:
    """
    Enhanced XGBoost hyperparameter optimizer focused on achieving R² > 0.87
    while controlling overfitting and beating the Enhanced Linear Regression baseline.
    """

    def __init__(self):
        self.model = None
        self.best_params = None
        self.feature_names = None
        self.performance_metrics = {}
        self.baseline_results = None
        self.optimization_history = []

    def load_baseline_results(self):
        """Load all previous results for comprehensive comparison."""
        print("=" * 70)
        print("ENHANCED XGBOOST HYPERPARAMETER OPTIMIZATION")
        print("=" * 70)
        print("Loading baseline results for strategic optimization...")

        # Load Enhanced Linear Regression results
        try:
            with open('results/models/enhanced_baseline_summary.json', 'r') as f:
                linear_results = json.load(f)
                linear_r2 = linear_results['performance_metrics']['test']['r2_score']
                print(f"✅ Enhanced Linear Regression R² = {linear_r2:.4f}")
        except FileNotFoundError:
            print("❌ Enhanced Linear Regression results not found!")
            return False

        # Load Enhanced XGBoost Baseline results
        try:
            with open('results/models/enhanced_xgboost_baseline_summary.json', 'r') as f:
                xgb_baseline = json.load(f)
                xgb_r2 = xgb_baseline['performance_metrics']['test']['r2_score']
                overfitting_gap = xgb_baseline['overfitting_analysis']['overfitting_gap']
                print(f"⚠️  Enhanced XGBoost Baseline R² = {xgb_r2:.4f}")
                print(f"⚠️  Overfitting gap = {overfitting_gap:.4f}")
        except FileNotFoundError:
            print("❌ Enhanced XGBoost baseline not found! Run baseline first.")
            return False

        self.baseline_results = {
            'linear_r2': linear_r2,
            'xgboost_baseline_r2': xgb_r2,
            'overfitting_gap': overfitting_gap,
            'performance_gap': linear_r2 - xgb_r2
        }

        print(f"\n🎯 Optimization Objectives (Priority Order):")
        print(f"   1. 🚨 CRITICAL: Reduce overfitting gap from {overfitting_gap:.4f} to <0.05")
        print(f"   2. 🏆 PRIMARY: Achieve thesis target R² > 0.87")
        print(f"   3. 📈 SECONDARY: Beat Linear Regression R² > {linear_r2:.4f}")
        print(f"   4. 🔧 STRATEGY: Aggressive regularization + careful parameter tuning")

        return True

    def load_and_prepare_data(self):
        """Load enhanced data with quality validation."""
        print("\n" + "=" * 50)
        print("DATA LOADING AND VALIDATION")
        print("=" * 50)

        # Load enhanced processed data
        df = pd.read_csv('data/processed/insurance_processed.csv')
        print(f"✅ Enhanced dataset loaded: {df.shape[0]} records, {df.shape[1]} features")

        # Validate enhanced features are present
        enhanced_features = ['high_risk', 'smoker_bmi_interaction', 'smoker_age_interaction', 'cost_complexity_score']
        available_enhanced = [f for f in enhanced_features if f in df.columns]
        print(f"✅ Enhanced features available: {len(available_enhanced)}/{len(enhanced_features)}")

        # Display feature correlations for optimization guidance
        print(f"\n📊 Key Features for Optimization:")
        for feature in available_enhanced:
            corr = df[feature].corr(df['charges'])
            print(f"   {feature}: r={corr:.3f}")

        return df

    def prepare_features_optimized(self, df):
        """Prepare features with optimization-focused encoding."""
        print("\n" + "=" * 50)
        print("OPTIMIZATION-FOCUSED FEATURE PREPARATION")
        print("=" * 50)

        df_opt = df.copy()

        # Label encoding for XGBoost efficiency
        label_encoders = {}
        categorical_features = ['sex', 'smoker', 'region', 'bmi_category', 'age_group']

        for feature in categorical_features:
            if feature in df_opt.columns:
                le = LabelEncoder()
                df_opt[feature] = le.fit_transform(df_opt[feature])
                label_encoders[feature] = le

        # Select all available features (enhanced preprocessing should have cleaned everything)
        exclude_features = ['charges', 'log_charges']
        feature_columns = [col for col in df_opt.columns if col not in exclude_features]

        X = df_opt[feature_columns]
        y = df_opt['charges']

        # Final data quality check for optimization
        print(f"✅ Optimization Data Quality:")
        print(f"   Features: {len(feature_columns)}")
        print(f"   Missing values: {X.isnull().sum().sum()}")
        print(f"   Target range: ${y.min():,.0f} - ${y.max():,.0f}")
        print(f"   Enhanced features: {len([f for f in feature_columns if f not in ['age', 'sex', 'bmi', 'children', 'smoker', 'region']])}")

        self.feature_names = feature_columns
        return X, y, label_encoders

    def split_data_consistent(self, X, y):
        """Split data consistently with all previous experiments."""
        print("\n" + "=" * 50)
        print("CONSISTENT DATA SPLITTING")
        print("=" * 50)

        # Use same random state and strategy as all previous experiments
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42,
            stratify=pd.qcut(y, q=5, duplicates='drop')
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42,
            stratify=pd.qcut(y_temp, q=3, duplicates='drop')
        )

        print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def define_anti_overfitting_search_space(self):
        """
        Define search space specifically designed to combat overfitting
        while achieving R² > 0.87 target.
        """
        print("\n" + "=" * 50)
        print("ANTI-OVERFITTING HYPERPARAMETER SEARCH SPACE")
        print("=" * 50)

        # Aggressive anti-overfitting parameter space
        if BAYESIAN_AVAILABLE:
            search_space = {
                # Learning rate: Lower values for better generalization
                'learning_rate': Real(0.01, 0.15, prior='log-uniform'),

                # Trees: More trees with lower learning rate
                'n_estimators': Integer(300, 1500),

                # Depth: Controlled depth to prevent overfitting
                'max_depth': Integer(3, 7),

                # Regularization: Strong regularization focus
                'reg_alpha': Real(0.1, 10.0, prior='log-uniform'),  # L1 regularization
                'reg_lambda': Real(1.0, 50.0, prior='log-uniform'), # L2 regularization

                # Child weight: Higher values prevent overfitting
                'min_child_weight': Integer(3, 20),

                # Sampling: Prevent overfitting through sampling
                'subsample': Real(0.6, 0.9),
                'colsample_bytree': Real(0.6, 0.9),
                'colsample_bylevel': Real(0.6, 0.9),

                # Gamma: Minimum split loss for regularization
                'gamma': Real(0.0, 5.0),
            }

            print(f"🚀 Using Bayesian Optimization with {len(search_space)} parameters")

        else:
            search_space = {
                # Learning rate: Lower values for better generalization
                'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],

                # Trees: More trees with lower learning rate
                'n_estimators': [400, 600, 800, 1000, 1200],

                # Depth: Controlled depth to prevent overfitting
                'max_depth': [3, 4, 5, 6],

                # Regularization: Strong regularization focus
                'reg_alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                'reg_lambda': [1.0, 2.0, 5.0, 10.0, 20.0, 30.0],

                # Child weight: Higher values prevent overfitting
                'min_child_weight': [3, 5, 7, 10, 15, 20],

                # Sampling: Prevent overfitting through sampling
                'subsample': [0.6, 0.7, 0.8, 0.9],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                'colsample_bylevel': [0.6, 0.7, 0.8, 0.9],

                # Gamma: Minimum split loss for regularization
                'gamma': [0.0, 0.1, 0.5, 1.0, 2.0],
            }

            total_combinations = np.prod([len(v) for v in search_space.values()])
            print(f"🔄 Using RandomizedSearchCV with {total_combinations:,} total combinations")

        print(f"\n🎯 Anti-Overfitting Strategy:")
        print(f"   🔧 Strong regularization: L1 (0.1-10.0) + L2 (1.0-50.0)")
        print(f"   🔧 Higher min_child_weight: 3-20 (vs baseline 1)")
        print(f"   🔧 Controlled depth: 3-7 (vs baseline 8)")
        print(f"   🔧 Aggressive sampling: 60-90% features/samples")
        print(f"   🔧 Lower learning rates: 0.01-0.15 (longer training)")
        print(f"   🎯 Target: Overfitting gap < 0.05, R² > 0.87")

        return search_space

    def optimize_xgboost_enhanced(self, X_train, y_train, X_val, y_val):
        """
        Perform enhanced hyperparameter optimization with overfitting focus.
        """
        print("\n" + "=" * 50)
        print("ENHANCED HYPERPARAMETER OPTIMIZATION")
        print("=" * 50)

        # Base parameters
        base_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
            'early_stopping_rounds': 50  # Early stopping for overfitting control
        }

        # Get search space
        search_space = self.define_anti_overfitting_search_space()

        # Create XGBoost regressor
        xgb_model = XGBRegressor(**base_params)

        print(f"\n🚀 Starting enhanced optimization...")
        optimization_start = time.time()

        if BAYESIAN_AVAILABLE:
            # Bayesian optimization
            print(f"🧠 Using Bayesian optimization for efficient search...")
            optimizer = BayesSearchCV(
                estimator=xgb_model,
                search_spaces=search_space,
                n_iter=100,  # More efficient than random search
                cv=5,
                scoring='r2',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
        else:
            # Random search with many iterations
            print(f"🎲 Using RandomizedSearchCV with comprehensive search...")
            optimizer = RandomizedSearchCV(
                estimator=xgb_model,
                param_distributions=search_space,
                n_iter=300,  # Comprehensive search
                cv=5,
                scoring='r2',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )

        # Perform optimization with early stopping
        try:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            optimizer.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        except Exception as e:
            print(f"⚠️  Falling back to standard fit...")
            optimizer.fit(X_train, y_train)

        optimization_time = time.time() - optimization_start

        # Extract results
        self.model = optimizer.best_estimator_
        self.best_params = optimizer.best_params_

        print(f"\n✅ Enhanced optimization completed in {optimization_time/60:.1f} minutes")
        print(f"🏆 Best CV R² Score: {optimizer.best_score_:.4f}")

        # Analyze improvement vs baseline
        baseline_r2 = self.baseline_results['xgboost_baseline_r2']
        cv_improvement = optimizer.best_score_ - baseline_r2
        print(f"📈 CV improvement vs baseline: {cv_improvement:+.4f}")

        print(f"\n🎯 Optimized Hyperparameters:")
        for param, value in self.best_params.items():
            print(f"   {param}: {value}")

        return optimization_time

    def evaluate_optimized_performance(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Comprehensive evaluation of optimized model."""
        print("\n" + "=" * 50)
        print("OPTIMIZED MODEL PERFORMANCE EVALUATION")
        print("=" * 50)

        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        y_test_pred = self.model.predict(X_test)

        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred, "Optimized Training")
        val_metrics = self._calculate_metrics(y_val, y_val_pred, "Optimized Validation")
        test_metrics = self._calculate_metrics(y_test, y_test_pred, "Optimized Test")

        # Store metrics
        self.performance_metrics = {
            'training': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }

        # CRITICAL: Overfitting analysis
        train_test_gap = train_metrics['r2_score'] - test_metrics['r2_score']
        baseline_gap = self.baseline_results['overfitting_gap']

        print(f"\n🔍 OVERFITTING CONTROL ANALYSIS:")
        print(f"   Baseline overfitting gap: {baseline_gap:.4f}")
        print(f"   Optimized overfitting gap: {train_test_gap:.4f}")
        print(f"   Gap reduction: {baseline_gap - train_test_gap:.4f}")

        if train_test_gap < 0.05:
            print(f"   ✅ EXCELLENT: Overfitting well controlled (gap < 0.05)")
        elif train_test_gap < 0.10:
            print(f"   ✅ GOOD: Overfitting controlled (gap < 0.10)")
        elif train_test_gap < baseline_gap:
            print(f"   📈 IMPROVED: Overfitting reduced vs baseline")
        else:
            print(f"   ⚠️  Overfitting still present - may need more regularization")

        return test_metrics

    def _calculate_metrics(self, y_true, y_pred, set_name):
        """Calculate comprehensive performance metrics."""
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

        print(f"\n{set_name} Performance:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: ${rmse:,.2f}")
        print(f"  MAE: ${mae:,.2f}")
        print(f"  MAPE: {mape:.2f}%")

        return metrics

    def comprehensive_model_comparison(self):
        """Compare optimized model with all previous models."""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE MODEL EVOLUTION COMPARISON")
        print("=" * 60)

        # Extract all model performances
        linear_r2 = self.baseline_results['linear_r2']
        xgb_baseline_r2 = self.baseline_results['xgboost_baseline_r2']
        optimized_r2 = self.performance_metrics['test']['r2_score']

        # Calculate improvements
        vs_linear = optimized_r2 - linear_r2
        vs_baseline = optimized_r2 - xgb_baseline_r2

        print(f"📊 Complete Model Evolution:")
        print(f"{'Model':<30} {'R² Score':<12} {'vs Linear':<12} {'vs Baseline':<12}")
        print("-" * 70)
        print(f"{'Enhanced Linear Regression':<30} {linear_r2:<12.4f} {'baseline':<12} {'-':<12}")
        print(f"{'Enhanced XGBoost Baseline':<30} {xgb_baseline_r2:<12.4f} {xgb_baseline_r2-linear_r2:+.4f} {'baseline':<12}")
        print(f"{'Enhanced XGBoost Optimized':<30} {optimized_r2:<12.4f} {vs_linear:+.4f} {vs_baseline:+.4f}")

        print(f"\n🎯 THESIS TARGET ACHIEVEMENT ANALYSIS:")

        # Thesis target (R² > 0.87)
        if optimized_r2 > 0.87:
            print(f"   🎉 THESIS TARGET ACHIEVED: R² = {optimized_r2:.4f} > 0.87")
            print(f"   🏆 CONGRATULATIONS! Enhanced XGBoost exceeds thesis requirement!")
            print(f"   🚀 Ready for Phase 5: Explainable AI (SHAP & LIME)!")
        else:
            gap_to_thesis = 0.87 - optimized_r2
            print(f"   📊 Current Performance: R² = {optimized_r2:.4f}")
            print(f"   🎯 Gap to thesis target: {gap_to_thesis:.4f}")
            if gap_to_thesis < 0.01:
                print(f"   ✅ Very close to target - consider ensemble methods!")
            else:
                print(f"   📈 Additional optimization strategies may be needed")

        # Beat Linear Regression target
        if vs_linear > 0:
            print(f"\n✅ XGBOOST SUPERIORITY ACHIEVED:")
            print(f"   📈 Beat Enhanced Linear Regression: +{vs_linear:.4f}")
            print(f"   🏆 XGBoost demonstrates advantage over linear methods!")
        else:
            print(f"\n⚠️  XGBoost vs Linear Regression:")
            print(f"   📊 Performance gap: {vs_linear:.4f}")
            print(f"   💡 Enhanced features may already capture most patterns linearly")

        # Optimization effectiveness
        if vs_baseline > 0:
            print(f"\n🔧 OPTIMIZATION EFFECTIVENESS:")
            print(f"   📈 Improvement vs baseline: +{vs_baseline:.4f}")
            print(f"   ✅ Hyperparameter optimization was successful!")
        else:
            print(f"\n⚠️  Optimization Results:")
            print(f"   📊 Change vs baseline: {vs_baseline:.4f}")
            print(f"   💡 May indicate optimal baseline parameters or overfitting control trade-off")

    def analyze_optimized_feature_importance(self):
        """Analyze feature importance in optimized model."""
        print("\n" + "=" * 50)
        print("OPTIMIZED MODEL FEATURE IMPORTANCE")
        print("=" * 50)

        # Get feature importance
        importance_types = ['weight', 'gain', 'cover']
        importance_data = {}

        for imp_type in importance_types:
            importance = self.model.get_booster().get_score(importance_type=imp_type)
            importance_data[imp_type] = importance

        # Create DataFrame
        feature_importance_df = pd.DataFrame(index=self.feature_names)
        for imp_type in importance_types:
            importance_values = [importance_data[imp_type].get(feature, 0) for feature in self.feature_names]
            feature_importance_df[imp_type] = importance_values

        feature_importance_df = feature_importance_df.sort_values('gain', ascending=False)

        print("Top 10 Features (by gain):")
        print(feature_importance_df.head(10))

        # Check if enhanced features still dominate after optimization
        enhanced_features = ['high_risk', 'smoker_bmi_interaction', 'smoker_age_interaction',
                           'cost_complexity_score', 'extreme_obesity', 'senior_smoker']

        enhanced_in_top_5 = len([f for f in feature_importance_df.head(5).index if f in enhanced_features])
        enhanced_in_top_10 = len([f for f in feature_importance_df.head(10).index if f in enhanced_features])

        print(f"\n📊 Enhanced Feature Impact After Optimization:")
        print(f"   Enhanced features in top 5: {enhanced_in_top_5}/5")
        print(f"   Enhanced features in top 10: {enhanced_in_top_10}/10")
        print(f"   Enhanced feature dominance: {enhanced_in_top_10/10*100:.1f}%")

        return feature_importance_df

    def create_optimization_visualizations(self, X_test, y_test, feature_importance_df):
        """Create comprehensive optimization result visualizations."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))

        # 1. Model Evolution Comparison
        models = ['Enhanced\nLinear', 'XGBoost\nBaseline', 'XGBoost\nOptimized']
        r2_scores = [
            self.baseline_results['linear_r2'],
            self.baseline_results['xgboost_baseline_r2'],
            self.performance_metrics['test']['r2_score']
        ]
        colors = ['blue', 'orange', 'green']

        bars = axes[0, 0].bar(models, r2_scores, color=colors, alpha=0.7)
        axes[0, 0].axhline(y=0.87, color='red', linestyle='--', label='Thesis Target (0.87)')
        axes[0, 0].axhline(y=0.86, color='purple', linestyle='--', label='Strong Performance')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('Model Evolution: Enhanced Data Optimization')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, score in zip(bars, r2_scores):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, score + 0.005,
                           f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

        # 2. Overfitting Control Comparison
        models_overfitting = ['Baseline', 'Optimized']
        overfitting_gaps = [
            self.baseline_results['overfitting_gap'],
            self.performance_metrics['training']['r2_score'] - self.performance_metrics['test']['r2_score']
        ]
        colors_overfitting = ['red', 'green']

        bars_overfitting = axes[0, 1].bar(models_overfitting, overfitting_gaps, color=colors_overfitting, alpha=0.7)
        axes[0, 1].axhline(y=0.05, color='orange', linestyle='--', label='Good Control (0.05)')
        axes[0, 1].axhline(y=0.10, color='red', linestyle='--', label='Acceptable (0.10)')
        axes[0, 1].set_ylabel('Overfitting Gap (Train R² - Test R²)')
        axes[0, 1].set_title('Overfitting Control: Baseline vs Optimized')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, gap in zip(bars_overfitting, overfitting_gaps):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, gap + 0.005,
                           f'{gap:.4f}', ha='center', va='bottom', fontweight='bold')

        # 3. Feature Importance (Top 8)
        top_features = feature_importance_df.head(8)
        axes[0, 2].barh(range(len(top_features)), top_features['gain'], color='lightgreen')
        axes[0, 2].set_yticks(range(len(top_features)))
        axes[0, 2].set_yticklabels(top_features.index)
        axes[0, 2].set_xlabel('Gain')
        axes[0, 2].set_title('Optimized Model Feature Importance')
        axes[0, 2].grid(axis='x', alpha=0.3)

        # 4. Prediction vs Actual
        y_test_pred = self.model.predict(X_test)
        axes[1, 0].scatter(y_test, y_test_pred, alpha=0.6, color='purple')
        axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                       'r--', lw=2, label='Perfect Prediction')
        axes[1, 0].set_xlabel('Actual Charges ($)')
        axes[1, 0].set_ylabel('Predicted Charges ($)')
        axes[1, 0].set_title('Optimized Model: Predicted vs Actual')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # Add R² annotation
        test_r2 = self.performance_metrics['test']['r2_score']
        axes[1, 0].text(0.05, 0.95, f'R² = {test_r2:.4f}', transform=axes[1, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 5. Residuals Analysis
        residuals = y_test - y_test_pred
        axes[1, 1].scatter(y_test_pred, residuals, alpha=0.6, color='red')
        axes[1, 1].axhline(y=0, color='black', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Charges ($)')
        axes[1, 1].set_ylabel('Residuals ($)')
        axes[1, 1].set_title('Optimized Model: Residuals Analysis')
        axes[1, 1].grid(alpha=0.3)

        # 6. Parameter Impact Visualization
        if self.best_params:
            # Show key optimized parameters
            key_params = ['learning_rate', 'max_depth', 'reg_alpha', 'reg_lambda', 'min_child_weight']
            param_values = []
            param_labels = []

            for param in key_params:
                if param in self.best_params:
                    param_values.append(self.best_params[param])
                    param_labels.append(param)

            if param_values:
                # Normalize values for visualization
                normalized_values = []
                for val in param_values:
                    if isinstance(val, (int, float)):
                        normalized_values.append(val)
                    else:
                        normalized_values.append(0)

                axes[1, 2].barh(range(len(param_labels)), normalized_values, color='lightcoral')
                axes[1, 2].set_yticks(range(len(param_labels)))
                axes[1, 2].set_yticklabels(param_labels)
                axes[1, 2].set_xlabel('Parameter Value')
                axes[1, 2].set_title('Key Optimized Parameters')
                axes[1, 2].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/plots/04_enhanced_xgboost_optimization.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✅ Optimization visualization saved: results/plots/04_enhanced_xgboost_optimization.png")

    def save_optimization_results(self, optimization_time):
        """Save comprehensive optimization results."""
        print("\n" + "=" * 50)
        print("SAVING OPTIMIZATION RESULTS")
        print("=" * 50)

        # Save optimized model
        model_path = 'results/models/enhanced_xgboost_optimized.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✅ Optimized model saved: {model_path}")

        # Create comprehensive summary
        summary = {
            'model_type': 'Enhanced XGBoost - Hyperparameter Optimized',
            'optimization_strategy': 'Anti-overfitting focused optimization with enhanced features',
            'optimization_method': 'Bayesian Optimization' if BAYESIAN_AVAILABLE else 'RandomizedSearchCV',
            'features_used': len(self.feature_names),
            'feature_names': self.feature_names,
            'optimized_hyperparameters': self.best_params,
            'performance_metrics': self.performance_metrics,
            'baseline_comparisons': {
                'enhanced_linear_r2': self.baseline_results['linear_r2'],
                'xgboost_baseline_r2': self.baseline_results['xgboost_baseline_r2'],
                'optimized_r2': self.performance_metrics['test']['r2_score'],
                'improvement_vs_linear': self.performance_metrics['test']['r2_score'] - self.baseline_results['linear_r2'],
                'improvement_vs_baseline': self.performance_metrics['test']['r2_score'] - self.baseline_results['xgboost_baseline_r2']
            },
            'overfitting_control': {
                'baseline_gap': self.baseline_results['overfitting_gap'],
                'optimized_gap': self.performance_metrics['training']['r2_score'] - self.performance_metrics['test']['r2_score'],
                'gap_reduction': self.baseline_results['overfitting_gap'] - (self.performance_metrics['training']['r2_score'] - self.performance_metrics['test']['r2_score']),
                'overfitting_controlled': (self.performance_metrics['training']['r2_score'] - self.performance_metrics['test']['r2_score']) < 0.10
            },
            'thesis_target_achievement': {
                'target_r2': 0.87,
                'achieved_r2': self.performance_metrics['test']['r2_score'],
                'target_met': self.performance_metrics['test']['r2_score'] >= 0.87,
                'gap_to_target': max(0, 0.87 - self.performance_metrics['test']['r2_score'])
            },
            'optimization_time_minutes': optimization_time / 60,
            'next_steps': 'Ready for Explainable AI (SHAP & LIME) if target achieved',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save summary
        summary_path = 'results/models/enhanced_xgboost_optimization_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=self._json_serialize)
        print(f"✅ Optimization summary saved: {summary_path}")

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
        print("ENHANCED XGBOOST OPTIMIZATION - FINAL SUMMARY")
        print("=" * 70)

        test_r2 = summary['performance_metrics']['test']['r2_score']
        targets = summary['thesis_target_achievement']
        overfitting = summary['overfitting_control']

        print(f"🚀 Enhanced XGBoost optimization completed!")
        print(f"📊 Final R² Score: {test_r2:.4f}")
        print(f"⏱️  Optimization time: {summary['optimization_time_minutes']:.1f} minutes")

        print(f"\n🎯 THESIS TARGET ACHIEVEMENT:")
        if targets['target_met']:
            print(f"   🎉 THESIS TARGET ACHIEVED: R² = {test_r2:.4f} ≥ 0.87")
            print(f"   🏆 CONGRATULATIONS! Enhanced XGBoost optimization successful!")
            print(f"   🚀 Ready for Phase 5: Explainable AI (SHAP & LIME)!")
        else:
            gap = targets['gap_to_target']
            print(f"   📊 Current Performance: R² = {test_r2:.4f}")
            print(f"   🎯 Gap to thesis target: {gap:.4f}")
            if gap < 0.01:
                print(f"   ✅ Very close to target - consider ensemble methods!")
            else:
                print(f"   📈 May need additional strategies (ensemble, feature engineering)")

        print(f"\n🔍 OVERFITTING CONTROL SUCCESS:")
        print(f"   Baseline overfitting gap: {overfitting['baseline_gap']:.4f}")
        print(f"   Optimized overfitting gap: {overfitting['optimized_gap']:.4f}")
        print(f"   Gap reduction: {overfitting['gap_reduction']:.4f}")

        if overfitting['overfitting_controlled']:
            print(f"   ✅ Overfitting successfully controlled!")
        else:
            print(f"   ⚠️  Some overfitting remains")

        comparisons = summary['baseline_comparisons']
        print(f"\n📈 MODEL COMPARISON RESULTS:")
        print(f"   vs Enhanced Linear Regression: {comparisons['improvement_vs_linear']:+.4f}")
        print(f"   vs XGBoost Baseline: {comparisons['improvement_vs_baseline']:+.4f}")

        if comparisons['improvement_vs_linear'] > 0:
            print(f"   ✅ XGBoost superiority demonstrated!")
        else:
            print(f"   📊 XGBoost comparable to Linear Regression")

        print(f"\n🔄 Next Steps:")
        if test_r2 >= 0.87:
            print(f"   🎯 READY FOR PHASE 5: Explainable AI Implementation")
            print(f"   📊 SHAP & LIME analysis with optimized model")
            print(f"   🎓 Dashboard development and thesis completion")
        else:
            print(f"   🔄 Consider ensemble methods or advanced techniques")
            print(f"   📊 Alternatively, proceed with current model if close to target")


def main():
    """Main execution function for enhanced XGBoost optimization."""
    # Create results directories
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/models', exist_ok=True)

    # Initialize optimizer
    optimizer = EnhancedXGBoostOptimizer()

    # Load baseline results for comparison
    if not optimizer.load_baseline_results():
        print("❌ Cannot proceed without baseline results!")
        return

    # Load and prepare data
    df = optimizer.load_and_prepare_data()
    X, y, label_encoders = optimizer.prepare_features_optimized(df)

    # Split data consistently
    X_train, X_val, X_test, y_train, y_val, y_test = optimizer.split_data_consistent(X, y)

    # Perform enhanced hyperparameter optimization
    optimization_time = optimizer.optimize_xgboost_enhanced(X_train, y_train, X_val, y_val)

    # Evaluate optimized model
    test_metrics = optimizer.evaluate_optimized_performance(X_train, y_train, X_val, y_val, X_test, y_test)

    # Comprehensive model comparison
    optimizer.comprehensive_model_comparison()

    # Analyze optimized feature importance
    feature_importance = optimizer.analyze_optimized_feature_importance()

    # Create optimization visualizations
    optimizer.create_optimization_visualizations(X_test, y_test, feature_importance)

    # Save optimization results
    summary = optimizer.save_optimization_results(optimization_time)

    print("\n" + "=" * 70)
    print("ENHANCED XGBOOST OPTIMIZATION COMPLETED")
    print("=" * 70)
    print("✅ Hyperparameter optimization completed")
    print("✅ Overfitting analysis and control applied")
    print("✅ Performance evaluation and comparison completed")
    print("✅ Optimized model ready for XAI implementation")

    final_r2 = test_metrics['r2_score']
    if final_r2 >= 0.87:
        print(f"\n🎉 THESIS SUCCESS: R² = {final_r2:.4f} ≥ 0.87 achieved!")
        print(f"🏆 Enhanced XGBoost optimization exceeds thesis target!")
        print(f"🚀 Ready for Phase 5: Explainable AI (SHAP & LIME)!")
    else:
        gap = 0.87 - final_r2
        print(f"\n📊 Final Performance: R² = {final_r2:.4f}")
        print(f"🎯 Gap to thesis target: {gap:.4f}")
        if gap < 0.01:
            print(f"✅ Very close to target - excellent progress!")

    print(f"\n📅 Project Status:")
    print(f"   ✅ Enhanced data preprocessing completed")
    print(f"   ✅ Enhanced Linear Regression baseline established")
    print(f"   ✅ XGBoost baseline and optimization completed")
    print(f"   🔄 Next: Phase 5 - Explainable AI implementation")


if __name__ == "__main__":
    main()