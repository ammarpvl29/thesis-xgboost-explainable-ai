"""
Phase 3: XGBoost Implementation with Hyperparameter Optimization
XGBoost Explainable AI for Patient Treatment Cost Prediction

Author: Ammar Pavel Zamora Siregar (1202224044)
Date: January 2025
Objective: Implement XGBoost model with comprehensive optimization for insurance cost prediction

This script implements Algorithm 3 from the thesis methodology - a comprehensive
XGBoost model with hyperparameter optimization, following the research objective
to achieve R² > 0.87 to demonstrate meaningful improvement over baseline (R² = 0.8637).

Following the thesis proposal methodology section 3.2.2:
"XGBoost Implementation untuk Healthcare Cost Prediction"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import os
import json
import pickle
from datetime import datetime
import time
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Set random seed for reproducibility
np.random.seed(42)

class XGBoostInsurancePrediction:
    """
    XGBoost implementation following Algorithm 3 from thesis methodology.
    Target: Achieve R² > 0.87 to demonstrate improvement over baseline (R² = 0.8637).
    """

    def __init__(self):
        self.model = None
        self.best_params = None
        self.feature_names = None
        self.performance_metrics = {}
        self.baseline_metrics = None
        self.training_history = []

    def load_baseline_results(self):
        """Load baseline results for comparison."""
        print("=" * 60)
        print("PHASE 3: XGBOOST IMPLEMENTATION - ALGORITHM 3")
        print("=" * 60)
        print("Loading baseline results for comparison...")

        try:
            with open('results/models/baseline_model_summary.json', 'r') as f:
                baseline_data = json.load(f)
                self.baseline_metrics = baseline_data['performance_metrics']

            baseline_r2 = self.baseline_metrics['test']['r2_score']
            print(f"✅ Baseline Linear Regression R² = {baseline_r2:.4f}")
            print(f"🎯 XGBoost Target: R² > 0.87 (improvement > {0.87 - baseline_r2:.4f})")

        except FileNotFoundError:
            print("⚠️  Baseline results not found. Run baseline model first!")
            return False

        return True

    def load_and_prepare_data(self):
        """Load processed data and prepare for XGBoost modeling."""
        print("\n" + "=" * 40)
        print("DATA LOADING AND PREPARATION")
        print("=" * 40)

        # Load processed data
        df = pd.read_csv('data/processed/insurance_processed.csv')
        print(f"Dataset loaded: {df.shape[0]} records, {df.shape[1]} features")

        # Handle any missing values
        if df['age_group'].isnull().sum() > 0:
            df['age_group'].fillna('18-29', inplace=True)
            print("Fixed missing age_group values")

        if df['bmi'].isnull().sum() > 0:
            median_bmi = df['bmi'].median()
            df['bmi'].fillna(median_bmi, inplace=True)
            print(f"Filled missing BMI values with median: {median_bmi:.2f}")

        print("\nDataset preview:")
        print(df[['age', 'bmi', 'smoker', 'high_risk', 'charges']].head())

        return df

    def prepare_features_for_xgboost(self, df):
        """
        Prepare features optimally for XGBoost.
        XGBoost handles categorical features natively, so we use different encoding.
        """
        print("\n" + "=" * 40)
        print("FEATURE PREPARATION FOR XGBOOST")
        print("=" * 40)

        df_xgb = df.copy()

        # For XGBoost, we can use label encoding for categorical features
        # This is more efficient than one-hot encoding for tree-based models
        label_encoders = {}

        # Encode binary categorical features
        binary_features = ['sex', 'smoker']
        for feature in binary_features:
            le = LabelEncoder()
            df_xgb[feature] = le.fit_transform(df_xgb[feature])
            label_encoders[feature] = le
            print(f"Label encoded {feature}: {list(le.classes_)}")

        # Encode multi-class categorical features
        categorical_features = ['region', 'bmi_category', 'age_group']
        for feature in categorical_features:
            if feature in df_xgb.columns:
                le = LabelEncoder()
                df_xgb[feature] = le.fit_transform(df_xgb[feature])
                label_encoders[feature] = le
                print(f"Label encoded {feature}: {list(le.classes_)}")

        # Select features for modeling (exclude target and derived features)
        exclude_features = ['charges', 'log_charges']
        feature_columns = [col for col in df_xgb.columns if col not in exclude_features]

        X = df_xgb[feature_columns]
        y = df_xgb['charges']  # Use original charges as target

        # Handle any remaining NaN values
        if X.isnull().sum().sum() > 0:
            print("WARNING: NaN values found, filling with 0")
            X = X.fillna(0)

        self.feature_names = feature_columns

        print(f"\nFinal XGBoost feature set: {len(feature_columns)} features")
        print(f"Features: {feature_columns}")
        print(f"Target: charges (${y.min():,.0f} - ${y.max():,.0f})")
        print(f"Final X shape: {X.shape}, Final y shape: {y.shape}")

        return X, y, label_encoders

    def split_data_consistent_with_baseline(self, X, y):
        """
        Split data using the same strategy as baseline for fair comparison.
        70% training, 15% validation, 15% testing with stratified sampling.
        """
        print("\n" + "=" * 40)
        print("DATA SPLITTING (CONSISTENT WITH BASELINE)")
        print("=" * 40)

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

        # Verify distribution preservation
        print(f"\nTarget distribution preservation:")
        print(f"Train mean: ${y_train.mean():,.0f}, std: ${y_train.std():,.0f}")
        print(f"Val mean: ${y_val.mean():,.0f}, std: ${y_val.std():,.0f}")
        print(f"Test mean: ${y_test.mean():,.0f}, std: ${y_test.std():,.0f}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_baseline_xgboost_params(self):
        """
        Get baseline XGBoost parameters without optimization.
        Using default/reasonable parameters for initial baseline performance.
        """
        print("\n" + "=" * 40)
        print("BASELINE XGBOOST PARAMETERS")
        print("=" * 40)

        # Baseline parameters - reasonable defaults for healthcare data
        baseline_params = {
            'n_estimators': 100,        # Standard number of trees
            'max_depth': 6,            # Default depth for XGBoost
            'learning_rate': 0.1,      # Default learning rate
            'subsample': 0.8,          # Slight regularization
            'colsample_bytree': 0.8,   # Slight feature sampling
            'reg_alpha': 0,            # No L1 regularization initially
            'reg_lambda': 1,           # Default L2 regularization
            'min_child_weight': 1,     # Default minimum child weight
            'gamma': 0,                # No minimum split loss initially
        }

        print(f"Baseline XGBoost parameters:")
        for param, value in baseline_params.items():
            print(f"  {param}: {value}")

        return baseline_params

    def train_baseline_xgboost(self, X_train, y_train, X_val, y_val):
        """
        Train baseline XGBoost model without hyperparameter optimization.
        Following professor's guidance to establish baseline performance first.
        """
        print("\n" + "=" * 40)
        print("BASELINE XGBOOST TRAINING (NO OPTIMIZATION)")
        print("=" * 40)

        # Get baseline parameters
        baseline_params = self.get_baseline_xgboost_params()

        # Base configuration parameters
        base_config = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',  # Faster for larger datasets
            'random_state': 42,
            'n_jobs': -1,  # Use all CPU cores
            'verbosity': 1  # Show some progress
        }

        # Combine baseline parameters with base configuration
        all_params = {**base_config, **baseline_params}

        # Create XGBoost regressor with baseline parameters
        self.model = XGBRegressor(**all_params)

        # Store baseline parameters for reporting
        self.best_params = baseline_params

        print(f"\nTraining baseline XGBoost model...")
        training_start = time.time()

        # Train the model with evaluation set for monitoring
        eval_set = [(X_train, y_train), (X_val, y_val)]

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False  # Reduce output during training
        )

        training_time = time.time() - training_start

        print(f"\n✅ Baseline XGBoost training completed in {training_time:.2f} seconds")

        # Cross-validation for baseline robustness assessment
        print(f"\nPerforming 5-fold cross-validation for baseline...")
        cv_start = time.time()
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='r2')
        cv_time = time.time() - cv_start

        print(f"Baseline 5-Fold CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Cross-validation completed in {cv_time:.2f} seconds")

        return training_time

    def calculate_metrics(self, y_true, y_pred, set_name):
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

        print(f"\n{set_name} Set Performance:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: ${rmse:,.2f}")
        print(f"  MAE: ${mae:,.2f}")
        print(f"  MAPE: {mape:.2f}%")

        return metrics

    def evaluate_model_performance(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Comprehensive model performance evaluation."""
        print("\n" + "=" * 40)
        print("COMPREHENSIVE PERFORMANCE EVALUATION")
        print("=" * 40)

        # Make predictions on all sets
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        y_test_pred = self.model.predict(X_test)

        # Calculate metrics for all sets
        train_metrics = self.calculate_metrics(y_train, y_train_pred, "Training")
        val_metrics = self.calculate_metrics(y_val, y_val_pred, "Validation")
        test_metrics = self.calculate_metrics(y_test, y_test_pred, "Test")

        # Store performance metrics
        self.performance_metrics = {
            'training': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }

        # Cross-validation for robustness
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='r2')
        print(f"\n5-Fold Cross-Validation R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return test_metrics

    def compare_with_baseline(self):
        """Compare XGBoost performance with baseline Linear Regression."""
        print("\n" + "=" * 60)
        print("PERFORMANCE COMPARISON: XGBOOST vs BASELINE")
        print("=" * 60)

        if self.baseline_metrics is None:
            print("❌ Baseline metrics not available for comparison")
            return

        # Extract key metrics
        baseline_r2 = self.baseline_metrics['test']['r2_score']
        baseline_rmse = self.baseline_metrics['test']['rmse']
        baseline_mae = self.baseline_metrics['test']['mae']

        xgb_r2 = self.performance_metrics['test']['r2_score']
        xgb_rmse = self.performance_metrics['test']['rmse']
        xgb_mae = self.performance_metrics['test']['mae']

        # Calculate improvements
        r2_improvement = xgb_r2 - baseline_r2
        rmse_improvement = ((baseline_rmse - xgb_rmse) / baseline_rmse) * 100
        mae_improvement = ((baseline_mae - xgb_mae) / baseline_mae) * 100

        print(f"📊 Performance Comparison Results:")
        print(f"{'Metric':<15} {'Baseline':<12} {'XGBoost':<12} {'Improvement':<15}")
        print("-" * 60)
        print(f"{'R² Score':<15} {baseline_r2:<12.4f} {xgb_r2:<12.4f} {r2_improvement:+.4f}")
        print(f"{'RMSE ($)':<15} {baseline_rmse:<12,.0f} {xgb_rmse:<12,.0f} {rmse_improvement:+.1f}%")
        print(f"{'MAE ($)':<15} {baseline_mae:<12,.0f} {xgb_mae:<12,.0f} {mae_improvement:+.1f}%")

        # Determine if target achieved
        target_r2 = 0.87
        if xgb_r2 >= target_r2:
            print(f"\n🎯 ✅ TARGET ACHIEVED: R² = {xgb_r2:.4f} ≥ {target_r2}")
            print(f"✅ XGBoost demonstrates meaningful improvement over baseline!")
        else:
            print(f"\n🎯 ⚠️  TARGET NOT ACHIEVED: R² = {xgb_r2:.4f} < {target_r2}")
            needed_improvement = target_r2 - xgb_r2
            print(f"⚠️  Need additional {needed_improvement:.4f} improvement to reach target")

        # Practical significance assessment
        if r2_improvement >= 0.01:
            print(f"✅ Practically significant improvement: +{r2_improvement:.4f} R²")
        else:
            print(f"⚠️  Limited practical improvement: +{r2_improvement:.4f} R²")

    def analyze_feature_importance(self):
        """Analyze XGBoost feature importance using multiple methods."""
        print("\n" + "=" * 40)
        print("XGBOOST FEATURE IMPORTANCE ANALYSIS")
        print("=" * 40)

        # Get different types of feature importance
        importance_types = ['weight', 'gain', 'cover']
        importance_data = {}

        for imp_type in importance_types:
            importance = self.model.get_booster().get_score(importance_type=imp_type)
            importance_data[imp_type] = importance

        # Create comprehensive feature importance DataFrame
        feature_importance_df = pd.DataFrame(index=self.feature_names)

        for imp_type in importance_types:
            importance_values = [importance_data[imp_type].get(feature, 0) for feature in self.feature_names]
            feature_importance_df[imp_type] = importance_values

        # Sort by gain (most informative for healthcare interpretation)
        feature_importance_df = feature_importance_df.sort_values('gain', ascending=False)

        print("Top 10 Most Important Features (by gain):")
        print(feature_importance_df.head(10).round(3))

        # Visualize feature importance
        self.visualize_feature_importance(feature_importance_df)

        return feature_importance_df

    def visualize_feature_importance(self, feature_importance_df):
        """Create comprehensive feature importance visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Feature importance by gain (top 10)
        top_features_gain = feature_importance_df.head(10)
        axes[0, 0].barh(range(len(top_features_gain)), top_features_gain['gain'], color='skyblue')
        axes[0, 0].set_yticks(range(len(top_features_gain)))
        axes[0, 0].set_yticklabels(top_features_gain.index)
        axes[0, 0].set_xlabel('Gain')
        axes[0, 0].set_title('Feature Importance by Gain (Top 10)')
        axes[0, 0].grid(axis='x', alpha=0.3)

        # 2. Feature importance by weight (frequency)
        top_features_weight = feature_importance_df.sort_values('weight', ascending=False).head(10)
        axes[0, 1].barh(range(len(top_features_weight)), top_features_weight['weight'], color='lightgreen')
        axes[0, 1].set_yticks(range(len(top_features_weight)))
        axes[0, 1].set_yticklabels(top_features_weight.index)
        axes[0, 1].set_xlabel('Weight (Frequency)')
        axes[0, 1].set_title('Feature Importance by Weight (Top 10)')
        axes[0, 1].grid(axis='x', alpha=0.3)

        # 3. Feature importance comparison - Gain vs Weight
        axes[1, 0].scatter(feature_importance_df['gain'], feature_importance_df['weight'], alpha=0.7)
        axes[1, 0].set_xlabel('Gain')
        axes[1, 0].set_ylabel('Weight')
        axes[1, 0].set_title('Feature Importance: Gain vs Weight')
        axes[1, 0].grid(alpha=0.3)

        # Add labels for top features
        for idx, (gain, weight, feature) in enumerate(zip(feature_importance_df['gain'][:5],
                                                          feature_importance_df['weight'][:5],
                                                          feature_importance_df.index[:5])):
            axes[1, 0].annotate(feature, (gain, weight), xytext=(5, 5),
                               textcoords='offset points', fontsize=8)

        # 4. Built-in XGBoost importance plot
        from xgboost import plot_importance
        plot_importance(self.model, ax=axes[1, 1], max_num_features=10, importance_type='gain')
        axes[1, 1].set_title('XGBoost Built-in Importance Plot')

        plt.tight_layout()
        plt.savefig('results/plots/10_xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_model_predictions(self, X_test, y_test):
        """Create comprehensive prediction visualization."""
        print("\n" + "=" * 40)
        print("PREDICTION VISUALIZATION")
        print("=" * 40)

        y_pred = self.model.predict(X_test)
        residuals = y_test - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Prediction vs Actual
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue')
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                        'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Charges ($)')
        axes[0, 0].set_ylabel('Predicted Charges ($)')
        axes[0, 0].set_title('XGBoost: Predicted vs Actual Charges')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # 2. Residuals plot
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Charges ($)')
        axes[0, 1].set_ylabel('Residuals ($)')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].grid(alpha=0.3)

        # 3. Residuals distribution
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_xlabel('Residuals ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residuals Distribution')
        axes[1, 0].axvline(residuals.mean(), color='red', linestyle='--',
                          label=f'Mean: ${residuals.mean():,.0f}')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # 4. Prediction error by actual value ranges
        # Bin actual values and calculate error metrics per bin
        bins = pd.qcut(y_test, q=5, labels=['Low', 'Low-Med', 'Medium', 'Med-High', 'High'])
        error_by_range = pd.DataFrame({
            'Range': bins,
            'Actual': y_test,
            'Predicted': y_pred,
            'AbsError': np.abs(residuals)
        })

        avg_error_by_range = error_by_range.groupby('Range')['AbsError'].mean()
        axes[1, 1].bar(avg_error_by_range.index, avg_error_by_range.values,
                       color='lightcoral', alpha=0.7)
        axes[1, 1].set_xlabel('Charge Range')
        axes[1, 1].set_ylabel('Mean Absolute Error ($)')
        axes[1, 1].set_title('Prediction Error by Charge Range')
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/plots/11_xgboost_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print error analysis
        print(f"Prediction Error Analysis:")
        print(f"Mean residual: ${residuals.mean():,.2f}")
        print(f"Std residual: ${residuals.std():,.2f}")
        print(f"Min residual: ${residuals.min():,.2f}")
        print(f"Max residual: ${residuals.max():,.2f}")

    def save_model_and_generate_summary(self, training_time):
        """Save baseline XGBoost model and generate comprehensive summary."""
        print("\n" + "=" * 40)
        print("BASELINE MODEL SAVING AND SUMMARY GENERATION")
        print("=" * 40)

        # Create models directory
        os.makedirs('results/models', exist_ok=True)

        # Save the trained baseline XGBoost model
        model_path = 'results/models/xgboost_baseline_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✅ Baseline XGBoost model saved to: {model_path}")

        # Generate comprehensive summary
        summary = {
            'model_type': 'XGBoost Regressor - Baseline (No Optimization)',
            'algorithm': 'Algorithm 3 - XGBoost Baseline Implementation',
            'optimization_status': 'baseline_only',
            'features_used': len(self.feature_names),
            'feature_names': self.feature_names,
            'baseline_parameters': self.best_params,
            'performance_metrics': self.performance_metrics,
            'linear_regression_comparison': self._create_baseline_comparison(),
            'training_time_seconds': training_time,
            'professor_target_achievement': self.performance_metrics['test']['r2_score'] > 0.86,
            'next_step': 'hyperparameter_optimization',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save summary as JSON
        summary_path = 'results/models/xgboost_baseline_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=self._json_serialize)
        print(f"✅ Baseline model summary saved to: {summary_path}")

        # Print summary
        self._print_final_summary(summary)

        return summary

    def _create_baseline_comparison(self):
        """Create detailed baseline comparison data."""
        if self.baseline_metrics is None:
            return None

        baseline_r2 = self.baseline_metrics['test']['r2_score']
        xgb_r2 = self.performance_metrics['test']['r2_score']

        return {
            'baseline_r2': baseline_r2,
            'xgboost_r2': xgb_r2,
            'improvement': xgb_r2 - baseline_r2,
            'relative_improvement_percent': ((xgb_r2 - baseline_r2) / baseline_r2) * 100,
            'target_r2': 0.87,
            'target_achieved': xgb_r2 >= 0.87
        }

    def _json_serialize(self, obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _print_final_summary(self, summary):
        """Print comprehensive final summary for baseline model."""
        print("\n" + "=" * 60)
        print("XGBOOST BASELINE IMPLEMENTATION SUMMARY")
        print("=" * 60)

        print(f"✅ Model Type: {summary['model_type']}")
        print(f"✅ Features Used: {summary['features_used']}")
        print(f"✅ Training Time: {summary['training_time_seconds']:.2f} seconds")

        test_metrics = summary['performance_metrics']['test']
        print(f"\n📊 Baseline XGBoost Test Performance:")
        print(f"   R² Score: {test_metrics['r2_score']:.4f}")
        print(f"   RMSE: ${test_metrics['rmse']:,.2f}")
        print(f"   MAE: ${test_metrics['mae']:,.2f}")
        print(f"   MAPE: {test_metrics['mape']:.2f}%")

        if summary['linear_regression_comparison']:
            comp = summary['linear_regression_comparison']
            print(f"\n📈 Comparison with Linear Regression Baseline:")
            print(f"   Linear Regression R²: {comp['baseline_r2']:.4f}")
            print(f"   Baseline XGBoost R²: {comp['xgboost_r2']:.4f}")
            print(f"   Improvement: +{comp['improvement']:.4f}")
            print(f"   Relative Improvement: +{comp['relative_improvement_percent']:.1f}%")

        # Professor's target check (> 0.86)
        if summary['professor_target_achievement']:
            print(f"\n🎯 ✅ PROFESSOR'S TARGET ACHIEVED: R² = {test_metrics['r2_score']:.4f} > 0.86")
        else:
            print(f"\n🎯 ⚠️  Professor's target not achieved: R² = {test_metrics['r2_score']:.4f} ≤ 0.86")

        print(f"\n📋 Baseline Parameters Used:")
        for param, value in summary['baseline_parameters'].items():
            print(f"   {param}: {value}")

        print(f"\n🔄 Next Steps:")
        print(f"   1. Hyperparameter optimization to improve beyond baseline")
        print(f"   2. Explainable AI integration (SHAP & LIME)")
        print(f"   3. Dashboard development")


def main():
    """Main execution function for XGBoost implementation."""
    # Create results directories
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/models', exist_ok=True)

    # Initialize XGBoost implementation
    xgb_insurance = XGBoostInsurancePrediction()

    # Load baseline results for comparison
    if not xgb_insurance.load_baseline_results():
        print("❌ Please run baseline model first!")
        return

    # Load and prepare data
    df = xgb_insurance.load_and_prepare_data()

    # Prepare features for XGBoost
    X, y, label_encoders = xgb_insurance.prepare_features_for_xgboost(df)

    # Split data consistently with baseline
    X_train, X_val, X_test, y_train, y_val, y_test = xgb_insurance.split_data_consistent_with_baseline(X, y)

    # Train baseline XGBoost model (no optimization)
    training_time = xgb_insurance.train_baseline_xgboost(X_train, y_train, X_val, y_val)

    # Evaluate model performance
    test_metrics = xgb_insurance.evaluate_model_performance(X_train, y_train, X_val, y_val, X_test, y_test)

    # Compare with baseline
    xgb_insurance.compare_with_baseline()

    # Analyze feature importance
    feature_importance = xgb_insurance.analyze_feature_importance()

    # Visualize predictions
    xgb_insurance.visualize_model_predictions(X_test, y_test)

    # Save baseline model and generate summary
    summary = xgb_insurance.save_model_and_generate_summary(training_time)

    print("\n" + "=" * 60)
    print("PHASE 3: BASELINE XGBOOST IMPLEMENTATION COMPLETED")
    print("=" * 60)
    print("✅ Baseline XGBoost model successfully implemented")
    print("✅ Performance evaluation and comparison completed")
    print("✅ Feature importance analysis completed")
    print("✅ Baseline model artifacts saved")
    print(f"\n📋 Professor's Timeline Progress:")
    print(f"   ✅ Baseline XGBoost training completed")
    print(f"   🔄 Next: Hyperparameter optimization (improve > 0.86)")
    print(f"   ⏳ Target: Complete optimization by September 30")
    print(f"\n🔄 Next Steps: Hyperparameter tuning, then XAI implementation")


if __name__ == "__main__":
    main()