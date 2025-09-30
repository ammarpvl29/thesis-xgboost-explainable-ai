"""
Phase 2 Enhanced: Baseline Linear Regression with Clean Data
XGBoost Explainable AI for Patient Treatment Cost Prediction

Author: Ammar Pavel Zamora Siregar (1202224044)
Date: September 2024
Objective: Establish enhanced baseline with clean, high-quality data (Target R² > 0.87)

This script implements Algorithm 2 from the thesis methodology using the enhanced
preprocessed data with 10.00/10.0 quality score. Expected significant improvement
over previous baseline (R² = 0.8637) due to:

1. Fixed data quality issues (missing values, BMI categories, age groups)
2. Corrected high_risk calculation (now r=0.815 with target)
3. Strategic interaction features based on healthcare domain knowledge
4. Enhanced feature engineering following medical standards

Target: Achieve R² > 0.87 with Linear Regression as strong baseline for XGBoost comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
import os
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12

# Set random seed for reproducibility
np.random.seed(42)

class EnhancedBaselineLinearRegression:
    """
    Enhanced Linear Regression baseline using high-quality preprocessed data.
    Expected to achieve R² > 0.87 due to improved data quality and feature engineering.
    """

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_names = None
        self.performance_metrics = {}
        self.feature_importance = None
        self.preprocessing_quality = None

    def load_enhanced_processed_data(self):
        """Load the enhanced preprocessed data with quality validation."""
        print("=" * 70)
        print("ENHANCED LINEAR REGRESSION BASELINE - ALGORITHM 2")
        print("=" * 70)
        print("Loading enhanced preprocessed data...")

        # Load enhanced processed data
        try:
            df = pd.read_csv('data/processed/insurance_processed.csv')
            print(f"✅ Enhanced dataset loaded: {df.shape[0]} records, {df.shape[1]} features")
        except FileNotFoundError:
            print("❌ Enhanced processed data not found!")
            print("Please run: python notebooks/00_enhanced_data_preprocessing.py")
            return None

        # Load preprocessing quality information
        try:
            with open('data/processed/preprocessing_enhanced_summary.json', 'r') as f:
                self.preprocessing_quality = json.load(f)
                quality_score = self.preprocessing_quality.get('quality_score', 0)
                print(f"✅ Data quality score: {quality_score}/10.0")
        except FileNotFoundError:
            print("⚠️  Quality summary not found, proceeding with data validation...")

        # Validate enhanced features
        expected_features = [
            'high_risk', 'smoker_bmi_interaction', 'smoker_age_interaction',
            'high_risk_age_interaction', 'extreme_obesity', 'senior_smoker',
            'cost_complexity_score'
        ]

        available_enhanced = [f for f in expected_features if f in df.columns]
        print(f"✅ Enhanced features available: {len(available_enhanced)}/{len(expected_features)}")
        print(f"   Features: {available_enhanced}")

        # Display correlation improvements
        print(f"\n📊 Key Feature Correlations with Target:")
        key_correlations = {
            'high_risk': df['high_risk'].corr(df['charges']) if 'high_risk' in df.columns else 0,
            'smoker': df['smoker'].map({'no': 0, 'yes': 1}).corr(df['charges']),
            'smoker_bmi_interaction': df['smoker_bmi_interaction'].corr(df['charges']) if 'smoker_bmi_interaction' in df.columns else 0,
            'cost_complexity_score': df['cost_complexity_score'].corr(df['charges']) if 'cost_complexity_score' in df.columns else 0
        }

        for feature, corr in sorted(key_correlations.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"   {feature}: {corr:.3f}")

        print(f"\nDataset preview:")
        print(df[['age', 'bmi', 'smoker', 'high_risk', 'charges']].head())

        return df

    def prepare_features_enhanced(self, df):
        """
        Prepare features using enhanced preprocessing with all improvements.
        Focus on leveraging the high-quality features for R² > 0.87 achievement.
        """
        print("\n" + "=" * 50)
        print("ENHANCED FEATURE PREPARATION")
        print("=" * 50)

        df_model = df.copy()

        # Encode categorical features for modeling
        print("🔧 Encoding categorical features:")
        label_encoders = {}
        categorical_features = ['sex', 'smoker', 'region', 'bmi_category', 'age_group']

        for feature in categorical_features:
            if feature in df_model.columns:
                le = LabelEncoder()
                df_model[feature] = le.fit_transform(df_model[feature])
                label_encoders[feature] = le
                print(f"   ✅ {feature}: {list(le.classes_)}")

        # Select features strategically
        print(f"\n📊 Strategic Feature Selection:")

        # Exclude target and log target
        exclude_features = ['charges', 'log_charges']

        # Include all enhanced features for maximum predictive power
        feature_columns = [col for col in df_model.columns if col not in exclude_features]

        X = df_model[feature_columns]
        y = df_model['charges']  # Use original charges for interpretability

        # Feature importance preview using correlation
        print(f"\n🎯 Feature Importance Preview (Correlation-based):")
        correlations = {}
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                corr = abs(X[col].corr(y))
                correlations[col] = corr

        # Display top 10 features by correlation
        top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]
        for feature, corr in top_features:
            print(f"   {feature}: {corr:.3f}")

        self.feature_names = feature_columns

        print(f"\n✅ Enhanced Feature Set Prepared:")
        print(f"   Total features: {len(feature_columns)}")
        print(f"   Original features: 6 (age, sex, bmi, children, smoker, region)")
        print(f"   Enhanced features: {len(feature_columns) - 6}")
        print(f"   Target: charges (${y.min():,.0f} - ${y.max():,.0f})")

        return X, y, label_encoders

    def split_data_enhanced(self, X, y):
        """
        Enhanced data splitting with stratification preservation.
        Maintains the same 70/15/15 strategy as thesis methodology.
        """
        print("\n" + "=" * 50)
        print("ENHANCED DATA SPLITTING STRATEGY")
        print("=" * 50)

        # First split: 70% train, 30% temp (maintaining stratification)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42,
            stratify=pd.qcut(y, q=5, duplicates='drop')
        )

        # Second split: 15% validation, 15% test from the 30%
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42,
            stratify=pd.qcut(y_temp, q=3, duplicates='drop')
        )

        print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

        # Verify enhanced feature distribution preservation
        print(f"\n📊 Target Distribution Preservation:")
        print(f"   Train mean: ${y_train.mean():,.0f}, std: ${y_train.std():,.0f}")
        print(f"   Val mean: ${y_val.mean():,.0f}, std: ${y_val.std():,.0f}")
        print(f"   Test mean: ${y_test.mean():,.0f}, std: ${y_test.std():,.0f}")

        # Check high-value cases distribution (important for healthcare costs)
        high_cost_threshold = y.quantile(0.95)
        train_high_cost = (y_train >= high_cost_threshold).sum()
        val_high_cost = (y_val >= high_cost_threshold).sum()
        test_high_cost = (y_test >= high_cost_threshold).sum()

        print(f"\n💰 High-Cost Cases Distribution (>95th percentile, ${high_cost_threshold:,.0f}):")
        print(f"   Train: {train_high_cost} ({train_high_cost/len(y_train)*100:.1f}%)")
        print(f"   Val: {val_high_cost} ({val_high_cost/len(y_val)*100:.1f}%)")
        print(f"   Test: {test_high_cost} ({test_high_cost/len(y_test)*100:.1f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_multiple_linear_models(self, X_train, y_train, X_val, y_val):
        """
        Train multiple linear regression variants to find the best performer.
        With enhanced features, we expect R² > 0.87 achievement.
        """
        print("\n" + "=" * 50)
        print("ENHANCED LINEAR REGRESSION TRAINING")
        print("=" * 50)

        # Define multiple linear regression variants
        models_config = {
            'linear_regression': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('linear', LinearRegression())
                ]),
                'description': 'Standard Linear Regression with scaling'
            },
            'ridge_regression': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('ridge', Ridge(alpha=1.0))
                ]),
                'description': 'Ridge Regression (L2 regularization)'
            },
            'lasso_regression': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('lasso', Lasso(alpha=0.1))
                ]),
                'description': 'Lasso Regression (L1 regularization)'
            },
            'elastic_net': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5))
                ]),
                'description': 'Elastic Net (L1 + L2 regularization)'
            }
        }

        print(f"🚀 Training {len(models_config)} linear regression variants:")

        model_results = {}

        for model_name, config in models_config.items():
            print(f"\n📊 Training {model_name}...")
            print(f"   Description: {config['description']}")

            start_time = datetime.now()

            # Train model
            model = config['model']
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, y_train_pred, f"{model_name} Training")
            val_metrics = self._calculate_metrics(y_val, y_val_pred, f"{model_name} Validation")

            # Cross-validation for robustness
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            training_time = (datetime.now() - start_time).total_seconds()

            model_results[model_name] = {
                'model': model,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'training_time': training_time,
                'description': config['description']
            }

            print(f"   ✅ Validation R²: {val_metrics['r2_score']:.4f}")
            print(f"   ✅ CV R²: {cv_mean:.4f} (±{cv_std:.4f})")
            print(f"   ⏱️  Training time: {training_time:.2f} seconds")

        # Select best model based on validation R²
        best_model_name = max(model_results.keys(),
                             key=lambda x: model_results[x]['val_metrics']['r2_score'])

        self.best_model = model_results[best_model_name]['model']
        self.models = model_results

        print(f"\n🏆 Best Model Selected: {best_model_name}")
        print(f"   Validation R²: {model_results[best_model_name]['val_metrics']['r2_score']:.4f}")
        print(f"   Description: {model_results[best_model_name]['description']}")

        return model_results, best_model_name

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

        # Only print for final test evaluation to reduce output
        if 'Test' in set_name:
            print(f"\n{set_name} Performance:")
            print(f"  R² Score: {r2:.4f}")
            print(f"  RMSE: ${rmse:,.2f}")
            print(f"  MAE: ${mae:,.2f}")
            print(f"  MAPE: {mape:.2f}%")

        return metrics

    def analyze_enhanced_feature_importance(self, X_train):
        """Analyze feature importance in the enhanced linear regression model."""
        print("\n" + "=" * 50)
        print("ENHANCED FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)

        # Get coefficients from the best model
        if hasattr(self.best_model.named_steps, 'linear'):
            coefficients = self.best_model.named_steps['linear'].coef_
        elif hasattr(self.best_model.named_steps, 'ridge'):
            coefficients = self.best_model.named_steps['ridge'].coef_
        elif hasattr(self.best_model.named_steps, 'lasso'):
            coefficients = self.best_model.named_steps['lasso'].coef_
        elif hasattr(self.best_model.named_steps, 'elastic'):
            coefficients = self.best_model.named_steps['elastic'].coef_
        else:
            coefficients = self.best_model.named_steps[-1].coef_

        # Create feature importance dataframe
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)

        print("Top 15 Most Important Features (by absolute coefficient):")
        print(self.feature_importance.head(15))

        # Categorize features
        print(f"\n📊 Feature Categories Analysis:")
        original_features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        enhanced_features = [f for f in self.feature_names if f not in original_features]

        original_in_top_10 = len([f for f in self.feature_importance.head(10)['feature'] if f in original_features])
        enhanced_in_top_10 = len([f for f in self.feature_importance.head(10)['feature'] if f in enhanced_features])

        print(f"   Original features in top 10: {original_in_top_10}")
        print(f"   Enhanced features in top 10: {enhanced_in_top_10}")
        print(f"   Enhancement impact: {enhanced_in_top_10/10*100:.1f}% of top features")

        return self.feature_importance

    def evaluate_final_performance(self, X_test, y_test):
        """Final evaluation on test set with enhanced model."""
        print("\n" + "=" * 50)
        print("ENHANCED MODEL FINAL EVALUATION")
        print("=" * 50)

        # Make predictions on test set
        y_test_pred = self.best_model.predict(X_test)

        # Calculate comprehensive test metrics
        test_metrics = self._calculate_metrics(y_test, y_test_pred, "Enhanced Linear Regression Test")

        # Compare with thesis target
        test_r2 = test_metrics['r2_score']
        print(f"\n🎯 Thesis Target Achievement Analysis:")
        if test_r2 >= 0.87:
            print(f"   🎉 THESIS TARGET ACHIEVED: R² = {test_r2:.4f} ≥ 0.87")
            print(f"   🏆 Enhanced Linear Regression exceeds thesis requirement!")
        elif test_r2 >= 0.86:
            print(f"   ✅ Strong Performance: R² = {test_r2:.4f} ≥ 0.86")
            print(f"   📈 Close to thesis target (gap: {0.87 - test_r2:.4f})")
        else:
            print(f"   📊 Current Performance: R² = {test_r2:.4f}")
            print(f"   🎯 Gap to thesis target: {0.87 - test_r2:.4f}")

        # Store final metrics
        self.performance_metrics = {
            'test': test_metrics,
            'model_type': 'Enhanced Linear Regression',
            'features_used': len(self.feature_names),
            'data_quality_score': self.preprocessing_quality.get('quality_score', 'Unknown') if self.preprocessing_quality else 'Unknown'
        }

        return test_metrics

    def create_enhanced_visualizations(self, X_test, y_test):
        """Create comprehensive visualizations of enhanced model performance."""
        print("\n" + "=" * 50)
        print("CREATING ENHANCED VISUALIZATIONS")
        print("=" * 50)

        y_test_pred = self.best_model.predict(X_test)
        residuals = y_test - y_test_pred

        fig, axes = plt.subplots(2, 3, figsize=(20, 14))

        # 1. Enhanced Model Performance Comparison
        model_names = list(self.models.keys())
        val_r2_scores = [self.models[name]['val_metrics']['r2_score'] for name in model_names]

        axes[0, 0].bar(model_names, val_r2_scores, color='skyblue', alpha=0.7)
        axes[0, 0].axhline(y=0.87, color='red', linestyle='--', label='Thesis Target (0.87)')
        axes[0, 0].axhline(y=0.86, color='orange', linestyle='--', label='Strong Performance (0.86)')
        axes[0, 0].set_title('Enhanced Linear Models Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Validation R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, score in enumerate(val_r2_scores):
            axes[0, 0].text(i, score + 0.005, f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

        # 2. Enhanced Feature Importance (Top 12)
        top_features = self.feature_importance.head(12)
        colors = ['red' if coef < 0 else 'blue' for coef in top_features['coefficient']]
        bars = axes[0, 1].barh(range(len(top_features)), top_features['coefficient'], color=colors, alpha=0.7)
        axes[0, 1].set_yticks(range(len(top_features)))
        axes[0, 1].set_yticklabels(top_features['feature'])
        axes[0, 1].set_xlabel('Coefficient Value')
        axes[0, 1].set_title('Enhanced Feature Importance (Top 12)', fontsize=14, fontweight='bold')
        axes[0, 1].grid(axis='x', alpha=0.3)

        # 3. Prediction vs Actual (Enhanced)
        axes[0, 2].scatter(y_test, y_test_pred, alpha=0.6, color='green')
        axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                       'r--', lw=2, label='Perfect Prediction')
        axes[0, 2].set_xlabel('Actual Charges ($)')
        axes[0, 2].set_ylabel('Predicted Charges ($)')
        axes[0, 2].set_title('Enhanced Model: Predicted vs Actual', fontsize=14, fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(alpha=0.3)

        # Add R² annotation
        test_r2 = self.performance_metrics['test']['r2_score']
        axes[0, 2].text(0.05, 0.95, f'R² = {test_r2:.4f}', transform=axes[0, 2].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=12, fontweight='bold')

        # 4. Residuals Analysis
        axes[1, 0].scatter(y_test_pred, residuals, alpha=0.6, color='orange')
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicted Charges ($)')
        axes[1, 0].set_ylabel('Residuals ($)')
        axes[1, 0].set_title('Enhanced Model: Residuals Analysis', fontsize=14, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)

        # 5. Residuals Distribution
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_xlabel('Residuals ($)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Residuals Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].axvline(residuals.mean(), color='red', linestyle='--',
                          label=f'Mean: ${residuals.mean():,.0f}')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        # 6. Model Evolution (if previous baseline exists)
        try:
            # Try to load previous baseline for comparison
            with open('results/models/baseline_model_summary.json', 'r') as f:
                previous_baseline = json.load(f)
                previous_r2 = previous_baseline['performance_metrics']['test']['r2_score']

                models = ['Previous\nBaseline', 'Enhanced\nBaseline']
                r2_scores = [previous_r2, test_r2]
                colors = ['lightblue', 'darkgreen']

                bars = axes[1, 2].bar(models, r2_scores, color=colors, alpha=0.7)
                axes[1, 2].axhline(y=0.87, color='red', linestyle='--', label='Thesis Target')
                axes[1, 2].set_ylabel('R² Score')
                axes[1, 2].set_title('Baseline Evolution', fontsize=14, fontweight='bold')
                axes[1, 2].legend()
                axes[1, 2].grid(axis='y', alpha=0.3)

                # Add improvement annotation
                improvement = test_r2 - previous_r2
                for i, (score, bar) in enumerate(zip(r2_scores, bars)):
                    axes[1, 2].text(bar.get_x() + bar.get_width()/2, score + 0.005,
                                   f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

                if improvement > 0:
                    axes[1, 2].text(0.5, 0.5, f'Improvement:\n+{improvement:.4f}',
                                   transform=axes[1, 2].transAxes, ha='center',
                                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        except FileNotFoundError:
            # If no previous baseline, show single performance
            axes[1, 2].bar(['Enhanced\nBaseline'], [test_r2], color='darkgreen', alpha=0.7)
            axes[1, 2].axhline(y=0.87, color='red', linestyle='--', label='Thesis Target')
            axes[1, 2].set_ylabel('R² Score')
            axes[1, 2].set_title('Enhanced Baseline Performance', fontsize=14, fontweight='bold')
            axes[1, 2].legend()
            axes[1, 2].text(0, test_r2 + 0.005, f'{test_r2:.4f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('results/plots/02_enhanced_baseline_performance.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✅ Enhanced visualization saved: results/plots/02_enhanced_baseline_performance.png")

    def save_enhanced_results(self):
        """Save enhanced baseline results with comprehensive summary."""
        print("\n" + "=" * 50)
        print("SAVING ENHANCED BASELINE RESULTS")
        print("=" * 50)

        # Create enhanced summary
        summary = {
            'model_type': 'Enhanced Linear Regression Baseline',
            'enhancement_strategy': 'High-quality data preprocessing + strategic feature engineering',
            'data_quality_score': self.preprocessing_quality.get('quality_score', 'Unknown') if self.preprocessing_quality else 'Unknown',
            'features_used': len(self.feature_names),
            'feature_names': self.feature_names,
            'best_model_type': [name for name, results in self.models.items()
                               if results['model'] == self.best_model][0],
            'all_models_performance': {
                name: {
                    'validation_r2': results['val_metrics']['r2_score'],
                    'cv_mean': results['cv_mean'],
                    'cv_std': results['cv_std'],
                    'description': results['description']
                }
                for name, results in self.models.items()
            },
            'performance_metrics': self.performance_metrics,
            'feature_importance_top_10': self.feature_importance.head(10).to_dict('records') if self.feature_importance is not None else [],
            'thesis_target_achievement': {
                'target_r2': 0.87,
                'achieved_r2': self.performance_metrics['test']['r2_score'],
                'target_met': self.performance_metrics['test']['r2_score'] >= 0.87,
                'gap_to_target': max(0, 0.87 - self.performance_metrics['test']['r2_score'])
            },
            'enhancement_impact': {
                'data_quality_improvements': 'Fixed missing values, BMI categories, age groups, high_risk calculation',
                'strategic_features_added': 'Smoker interactions, risk stratification, cost complexity scoring',
                'expected_xgboost_improvement': 'Enhanced baseline provides strong foundation for XGBoost optimization'
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save summary
        os.makedirs('results/models', exist_ok=True)
        summary_path = 'results/models/enhanced_baseline_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=self._json_serialize)
        print(f"✅ Enhanced baseline summary saved: {summary_path}")

        # Print final summary
        self._print_final_summary(summary)

        return summary

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
        """Print comprehensive final summary."""
        print("\n" + "=" * 70)
        print("ENHANCED LINEAR REGRESSION BASELINE - FINAL SUMMARY")
        print("=" * 70)

        test_r2 = summary['performance_metrics']['test']['r2_score']
        target_achievement = summary['thesis_target_achievement']

        print(f"✅ Enhanced baseline training completed successfully!")
        print(f"📊 Final R² Score: {test_r2:.4f}")
        print(f"🏆 Best Model: {summary['best_model_type']}")
        print(f"📈 Data Quality Score: {summary['data_quality_score']}/10.0")

        print(f"\n🎯 THESIS TARGET ACHIEVEMENT:")
        if target_achievement['target_met']:
            print(f"   🎉 THESIS TARGET ACHIEVED: R² = {test_r2:.4f} ≥ 0.87")
            print(f"   🏆 Enhanced Linear Regression exceeds thesis requirement!")
            print(f"   🚀 Excellent foundation for XGBoost optimization!")
        else:
            gap = target_achievement['gap_to_target']
            print(f"   📊 Current Performance: R² = {test_r2:.4f}")
            print(f"   🎯 Gap to thesis target: {gap:.4f}")
            if gap < 0.01:
                print(f"   ✅ Very close to target - XGBoost should easily exceed 0.87!")
            elif gap < 0.02:
                print(f"   ✅ Close to target - XGBoost optimization should achieve 0.87!")
            else:
                print(f"   📈 Solid baseline - XGBoost with optimization should achieve 0.87!")

        print(f"\n📊 Model Performance Summary:")
        test_metrics = summary['performance_metrics']['test']
        print(f"   R² Score: {test_metrics['r2_score']:.4f}")
        print(f"   RMSE: ${test_metrics['rmse']:,.2f}")
        print(f"   MAE: ${test_metrics['mae']:,.2f}")
        print(f"   MAPE: {test_metrics['mape']:.2f}%")

        print(f"\n🧬 Enhancement Impact:")
        print(f"   Features used: {summary['features_used']} (enhanced from 6 original)")
        print(f"   Data quality: {summary['data_quality_score']}/10.0")
        print(f"   Strategic enhancements: high_risk (r=0.815), interactions, risk stratification")

        print(f"\n🔄 Next Steps:")
        if test_r2 >= 0.87:
            print(f"   🎯 Target achieved with Linear Regression!")
            print(f"   🚀 XGBoost should significantly exceed 0.87")
            print(f"   📊 Proceed with confidence to Phase 3 optimization")
        else:
            print(f"   ✅ Strong baseline established ({test_r2:.4f})")
            print(f"   🚀 XGBoost optimization should achieve thesis target")
            print(f"   📊 Enhanced features provide excellent foundation")

        print(f"   🎓 Phase 4: Explainable AI (SHAP & LIME) implementation")
        print(f"   📊 Dashboard development with enhanced model")


def main():
    """Main execution function for enhanced baseline linear regression."""
    # Create results directory
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/models', exist_ok=True)

    # Initialize enhanced baseline
    baseline = EnhancedBaselineLinearRegression()

    # Load enhanced processed data
    df = baseline.load_enhanced_processed_data()
    if df is None:
        return

    # Prepare enhanced features
    X, y, label_encoders = baseline.prepare_features_enhanced(df)

    # Split data with enhanced strategy
    X_train, X_val, X_test, y_train, y_val, y_test = baseline.split_data_enhanced(X, y)

    # Train multiple linear models
    model_results, best_model_name = baseline.train_multiple_linear_models(X_train, y_train, X_val, y_val)

    # Analyze enhanced feature importance
    feature_importance = baseline.analyze_enhanced_feature_importance(X_train)

    # Final evaluation
    test_metrics = baseline.evaluate_final_performance(X_test, y_test)

    # Create enhanced visualizations
    baseline.create_enhanced_visualizations(X_test, y_test)

    # Save enhanced results
    summary = baseline.save_enhanced_results()

    print("\n" + "=" * 70)
    print("ENHANCED LINEAR REGRESSION BASELINE COMPLETED")
    print("=" * 70)
    print("✅ Enhanced preprocessing + strategic features applied")
    print("✅ Multiple linear models trained and optimized")
    print("✅ Comprehensive performance evaluation completed")
    print("✅ Ready for XGBoost optimization with strong baseline")

    final_r2 = test_metrics['r2_score']
    if final_r2 >= 0.87:
        print(f"\n🎉 EXCEPTIONAL: R² = {final_r2:.4f} ≥ 0.87 with Linear Regression!")
        print(f"🏆 Thesis target achieved with baseline - XGBoost will exceed expectations!")
    elif final_r2 >= 0.86:
        print(f"\n✅ EXCELLENT: R² = {final_r2:.4f} ≥ 0.86")
        print(f"🚀 Strong foundation for XGBoost to achieve thesis target!")
    else:
        print(f"\n📊 SOLID BASELINE: R² = {final_r2:.4f}")
        print(f"🎯 XGBoost optimization should achieve thesis target R² > 0.87!")


if __name__ == "__main__":
    main()