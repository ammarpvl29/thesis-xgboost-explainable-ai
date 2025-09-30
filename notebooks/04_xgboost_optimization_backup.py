import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, validation_curve
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import json
import pickle
from datetime import datetime
import time
from scipy import stats
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

np.random.seed(42)

class XGBoostOptimizer:
    """
    XGBoost hyperparameter optimization focused on reducing overfitting
    and achieving professor's target R² > 0.86.
    """

    def __init__(self):
        self.model = None
        self.best_params = None
        self.feature_names = None
        self.performance_metrics = {}
        self.baseline_results = None
        self.optimization_history = []

    def load_baseline_results(self):
        """Load baseline results for comparison."""
        print("=" * 60)
        print("PHASE 3B: XGBOOST HYPERPARAMETER OPTIMIZATION")
        print("=" * 60)
        print("Loading baseline results for strategic optimization...")

        try:
            with open('results/models/baseline_model_summary.json', 'r') as f:
                linear_baseline = json.load(f)
                linear_r2 = linear_baseline['performance_metrics']['test']['r2_score']
                print(f"✅ Linear Regression R² = {linear_r2:.4f}")
        except FileNotFoundError:
            print("❌ Linear Regression baseline not found!")
            return False

        try:
            with open('results/models/xgboost_baseline_summary.json', 'r') as f:
                xgb_baseline = json.load(f)
                xgb_r2 = xgb_baseline['performance_metrics']['test']['r2_score']
                print(f"⚠️  XGBoost Baseline R² = {xgb_r2:.4f}")

                self.baseline_results = {
                    'linear_r2': linear_r2,
                    'xgboost_baseline_r2': xgb_r2,
                    'performance_gap': linear_r2 - xgb_r2
                }

        except FileNotFoundError:
            print("❌ XGBoost baseline not found! Run baseline script first.")
            return False

        print(f"\n🎯 Optimization Targets:")
        print(f"   Professor's Target: R² > 0.86")
        print(f"   Beat Linear Regression: R² > {linear_r2:.4f}")
        print(f"   Current Gap to Close: {linear_r2 - xgb_r2:.4f}")
        print(f"   Focus: Reduce overfitting and improve generalization")

        return True

    def load_and_prepare_data(self):
        """Load processed data and prepare for optimization."""
        print("\n" + "=" * 40)
        print("DATA LOADING AND PREPARATION")
        print("=" * 40)

        df = pd.read_csv('data/processed/insurance_processed.csv')
        print(f"Dataset loaded: {df.shape[0]} records, {df.shape[1]} features")

        if df['age_group'].isnull().sum() > 0:
            df['age_group'].fillna('18-29', inplace=True)
            print("Fixed missing age_group values")

        if df['bmi'].isnull().sum() > 0:
            median_bmi = df['bmi'].median()
            df['bmi'].fillna(median_bmi, inplace=True)
            print(f"Filled missing BMI values with median: {median_bmi:.2f}")

        return df

    def prepare_features_for_optimization(self, df):
        """Prepare features for XGBoost optimization."""
        print("\n" + "=" * 40)
        print("FEATURE PREPARATION FOR OPTIMIZATION")
        print("=" * 40)

        df_xgb = df.copy()

        label_encoders = {}
        categorical_features = ['sex', 'smoker', 'region', 'bmi_category', 'age_group']

        for feature in categorical_features:
            if feature in df_xgb.columns:
                le = LabelEncoder()
                df_xgb[feature] = le.fit_transform(df_xgb[feature])
                label_encoders[feature] = le
                print(f"Label encoded {feature}: {list(le.classes_)}")

        exclude_features = ['charges', 'log_charges']
        feature_columns = [col for col in df_xgb.columns if col not in exclude_features]

        X = df_xgb[feature_columns]
        y = df_xgb['charges']

        if X.isnull().sum().sum() > 0:
            X = X.fillna(0)
            print("Filled remaining NaN values with 0")

        self.feature_names = feature_columns

        print(f"\nOptimization feature set: {len(feature_columns)} features")
        print(f"Features: {feature_columns}")
        print(f"Target: charges (${y.min():,.0f} - ${y.max():,.0f})")

        return X, y, label_encoders

    def split_data_consistent(self, X, y):
        """Split data using the same strategy as baseline for fair comparison."""
        from sklearn.model_selection import train_test_split

        print("\n" + "=" * 40)
        print("DATA SPLITTING (CONSISTENT WITH BASELINE)")
        print("=" * 40)

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=pd.qcut(y, q=5, duplicates='drop')
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42,
            stratify=pd.qcut(y_temp, q=3, duplicates='drop')
        )

        print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def define_focused_search_space(self):
        """
        Define focused hyperparameter search space based on baseline analysis.
        Prioritize regularization to reduce overfitting.
        """
        print("\n" + "=" * 40)
        print("FOCUSED HYPERPARAMETER SEARCH SPACE")
        print("=" * 40)

        param_grid = {
            'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],

            'n_estimators': [100, 200, 300, 500],

            'max_depth': [3, 4, 5, 6],
            'min_child_weight': [3, 5, 7, 10, 15],

            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],

            'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0, 2.0],
            'reg_lambda': [1.0, 2.0, 5.0, 10.0, 20.0],

            'gamma': [0, 0.1, 0.5, 1.0, 2.0, 5.0],
        }

        total_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"Total parameter combinations: {total_combinations:,}")
        print(f"Optimization Strategy:")
        print(f"  🎯 Focus: Regularization to reduce overfitting")
        print(f"  📊 Target: R² > 0.86 (professor's requirement)")
        print(f"  🔄 Method: RandomizedSearchCV with 300 iterations")
        print(f"  ⏱️  Expected time: 10-15 minutes")

        for param, values in param_grid.items():
            print(f"  {param}: {values}")

        return param_grid

    def optimize_xgboost(self, X_train, y_train, X_val, y_val):
        """
        Perform systematic hyperparameter optimization focused on reducing overfitting.
        """
        print("\n" + "=" * 40)
        print("SYSTEMATIC HYPERPARAMETER OPTIMIZATION")
        print("=" * 40)

        base_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }

        xgb_model = XGBRegressor(**base_params)

        param_grid = self.define_focused_search_space()

        print(f"\nStarting hyperparameter optimization...")
        print(f"Using 5-fold cross-validation with 300 parameter combinations")

        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=300,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1,
            random_state=42,
            return_train_score=True
        )

        optimization_start = time.time()

        print(f"🔄 Optimization in progress (this will take ~10-15 minutes)...")
        random_search.fit(X_train, y_train)

        optimization_time = time.time() - optimization_start

        print(f"\n✅ Optimization completed in {optimization_time/60:.1f} minutes")

        self.model = random_search.best_estimator_
        self.best_params = random_search.best_params_

        self.optimization_history = {
            'best_cv_score': random_search.best_score_,
            'cv_results': random_search.cv_results_,
            'optimization_time': optimization_time
        }

        print(f"\n🎯 Best Hyperparameters Found:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")

        print(f"\nBest CV R² Score: {random_search.best_score_:.4f}")

        return optimization_time

    def calculate_comprehensive_metrics(self, y_true, y_pred, set_name):
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

    def evaluate_optimized_model(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Comprehensive evaluation of optimized model."""
        print("\n" + "=" * 40)
        print("OPTIMIZED MODEL EVALUATION")
        print("=" * 40)

        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        y_test_pred = self.model.predict(X_test)

        train_metrics = self.calculate_comprehensive_metrics(y_train, y_train_pred, "Training")
        val_metrics = self.calculate_comprehensive_metrics(y_val, y_val_pred, "Validation")
        test_metrics = self.calculate_comprehensive_metrics(y_test, y_test_pred, "Test")

        self.performance_metrics = {
            'training': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }

        train_test_gap = train_metrics['r2_score'] - test_metrics['r2_score']
        print(f"\n🔍 Overfitting Analysis:")
        print(f"  Training R²: {train_metrics['r2_score']:.4f}")
        print(f"  Test R²: {test_metrics['r2_score']:.4f}")
        print(f"  Gap: {train_test_gap:.4f}")

        if train_test_gap < 0.05:
            print(f"  ✅ Good generalization (gap < 0.05)")
        elif train_test_gap < 0.10:
            print(f"  ⚠️  Moderate overfitting (gap 0.05-0.10)")
        else:
            print(f"  🚨 Significant overfitting (gap > 0.10)")

        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='r2')
        print(f"\n5-Fold CV R² Score: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")

        return test_metrics

    def compare_with_all_baselines(self):
        """Compare optimized XGBoost with all baseline models."""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE PERFORMANCE COMPARISON")
        print("=" * 60)

        if self.baseline_results is None:
            print("❌ Baseline results not available for comparison")
            return

        linear_r2 = self.baseline_results['linear_r2']
        xgb_baseline_r2 = self.baseline_results['xgboost_baseline_r2']
        optimized_r2 = self.performance_metrics['test']['r2_score']

        vs_linear = optimized_r2 - linear_r2
        vs_xgb_baseline = optimized_r2 - xgb_baseline_r2

        print(f"📊 Model Performance Comparison:")
        print(f"{'Model':<25} {'R² Score':<12} {'vs Linear':<12} {'vs XGB Base':<15}")
        print("-" * 70)
        print(f"{'Linear Regression':<25} {linear_r2:<12.4f} {'baseline':<12} {'-':<15}")
        print(f"{'XGBoost Baseline':<25} {xgb_baseline_r2:<12.4f} {xgb_baseline_r2-linear_r2:+.4f} {'baseline':<15}")
        print(f"{'XGBoost Optimized':<25} {optimized_r2:<12.4f} {vs_linear:+.4f} {vs_xgb_baseline:+.4f}")

        print(f"\n🎯 Target Achievement Analysis:")

        if optimized_r2 > 0.86:
            print(f"  ✅ PROFESSOR'S TARGET ACHIEVED: R² = {optimized_r2:.4f} > 0.86")
        else:
            gap_to_target = 0.86 - optimized_r2
            print(f"  ❌ Professor's target missed: R² = {optimized_r2:.4f} < 0.86")
            print(f"     Need additional {gap_to_target:.4f} improvement")

        if optimized_r2 > 0.87:
            print(f"  ✅ THESIS TARGET ACHIEVED: R² = {optimized_r2:.4f} > 0.87")
        else:
            gap_to_thesis = 0.87 - optimized_r2
            print(f"  ⚠️  Thesis target not achieved: R² = {optimized_r2:.4f} < 0.87")
            print(f"     Need additional {gap_to_thesis:.4f} improvement")

        if vs_linear > 0:
            print(f"  ✅ BEAT LINEAR REGRESSION: +{vs_linear:.4f} improvement")
        else:
            print(f"  ❌ Did not beat Linear Regression: {vs_linear:.4f} behind")

        if vs_linear >= 0.01:
            print(f"  ✅ Practically significant improvement over Linear Regression")
        else:
            print(f"  ⚠️  Limited practical improvement over Linear Regression")

    def analyze_optimized_feature_importance(self):
        """Analyze feature importance in optimized model."""
        print("\n" + "=" * 40)
        print("OPTIMIZED MODEL FEATURE IMPORTANCE")
        print("=" * 40)

        importance_types = ['weight', 'gain', 'cover']
        importance_data = {}

        for imp_type in importance_types:
            importance = self.model.get_booster().get_score(importance_type=imp_type)
            importance_data[imp_type] = importance

        feature_importance_df = pd.DataFrame(index=self.feature_names)
        for imp_type in importance_types:
            importance_values = [importance_data[imp_type].get(feature, 0) for feature in self.feature_names]
            feature_importance_df[imp_type] = importance_values

        feature_importance_df = feature_importance_df.sort_values('gain', ascending=False)

        print("Top 10 Features (by gain):")
        print(feature_importance_df.head(10).round(2))

        self.visualize_optimization_results(feature_importance_df)

        return feature_importance_df

    def visualize_optimization_results(self, feature_importance_df):
        """Create comprehensive visualizations of optimization results."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        top_features = feature_importance_df.head(8)
        axes[0, 0].barh(range(len(top_features)), top_features['gain'], color='skyblue')
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels(top_features.index)
        axes[0, 0].set_xlabel('Gain')
        axes[0, 0].set_title('Optimized XGBoost Feature Importance')
        axes[0, 0].grid(axis='x', alpha=0.3)

        models = ['Linear\nRegression', 'XGBoost\nBaseline', 'XGBoost\nOptimized']
        r2_scores = [
            self.baseline_results['linear_r2'],
            self.baseline_results['xgboost_baseline_r2'],
            self.performance_metrics['test']['r2_score']
        ]
        colors = ['blue', 'orange', 'green']

        bars = axes[0, 1].bar(models, r2_scores, color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_title('Model Performance Comparison')
        axes[0, 1].axhline(y=0.86, color='red', linestyle='--', label="Professor's Target")
        axes[0, 1].axhline(y=0.87, color='purple', linestyle='--', label='Thesis Target')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)

        for bar, score in zip(bars, r2_scores):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, score + 0.005,
                           f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

        y_test_pred = self.model.predict(self.X_test) if hasattr(self, 'X_test') else None
        if y_test_pred is not None and hasattr(self, 'y_test'):
            axes[1, 0].scatter(self.y_test, y_test_pred, alpha=0.6, color='green')
            axes[1, 0].plot([self.y_test.min(), self.y_test.max()],
                           [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            axes[1, 0].set_xlabel('Actual Charges ($)')
            axes[1, 0].set_ylabel('Predicted Charges ($)')
            axes[1, 0].set_title('Optimized Model: Predicted vs Actual')
            axes[1, 0].grid(alpha=0.3)

        if self.best_params:
            param_names = list(self.best_params.keys())[:6]
            param_values = [self.best_params[name] for name in param_names]

            param_values_norm = []
            for val in param_values:
                if isinstance(val, (int, float)):
                    param_values_norm.append(val)
                else:
                    param_values_norm.append(0)

            axes[1, 1].barh(range(len(param_names)), param_values_norm, color='lightcoral')
            axes[1, 1].set_yticks(range(len(param_names)))
            axes[1, 1].set_yticklabels(param_names)
            axes[1, 1].set_xlabel('Parameter Value')
            axes[1, 1].set_title('Optimized Hyperparameters')
            axes[1, 1].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/plots/12_xgboost_optimization_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_optimized_model(self, optimization_time):
        """Save optimized model and generate comprehensive summary."""
        print("\n" + "=" * 40)
        print("SAVING OPTIMIZED MODEL AND RESULTS")
        print("=" * 40)

        os.makedirs('results/models', exist_ok=True)
        model_path = 'results/models/xgboost_optimized_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✅ Optimized XGBoost model saved: {model_path}")

        summary = {
            'model_type': 'XGBoost Regressor - Hyperparameter Optimized',
            'optimization_strategy': 'Focused regularization to reduce overfitting',
            'optimization_method': 'RandomizedSearchCV with 300 iterations',
            'features_used': len(self.feature_names),
            'feature_names': self.feature_names,
            'best_hyperparameters': self.best_params,
            'performance_metrics': self.performance_metrics,
            'baseline_comparison': {
                'linear_regression_r2': self.baseline_results['linear_r2'],
                'xgboost_baseline_r2': self.baseline_results['xgboost_baseline_r2'],
                'optimized_xgboost_r2': self.performance_metrics['test']['r2_score'],
                'improvement_vs_linear': self.performance_metrics['test']['r2_score'] - self.baseline_results['linear_r2'],
                'improvement_vs_baseline': self.performance_metrics['test']['r2_score'] - self.baseline_results['xgboost_baseline_r2']
            },
            'target_achievement': {
                'professor_target_0_86': self.performance_metrics['test']['r2_score'] > 0.86,
                'thesis_target_0_87': self.performance_metrics['test']['r2_score'] > 0.87,
                'beat_linear_regression': self.performance_metrics['test']['r2_score'] > self.baseline_results['linear_r2']
            },
            'optimization_time_minutes': optimization_time / 60,
            'overfitting_analysis': {
                'train_r2': self.performance_metrics['training']['r2_score'],
                'test_r2': self.performance_metrics['test']['r2_score'],
                'overfitting_gap': self.performance_metrics['training']['r2_score'] - self.performance_metrics['test']['r2_score']
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        summary_path = 'results/models/xgboost_optimized_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=self._json_serialize)
        print(f"✅ Optimization summary saved: {summary_path}")

        self._print_optimization_summary(summary)

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

    def _print_optimization_summary(self, summary):
        """Print comprehensive optimization summary."""
        print("\n" + "=" * 60)
        print("XGBOOST OPTIMIZATION SUMMARY")
        print("=" * 60)

        test_r2 = summary['performance_metrics']['test']['r2_score']
        targets = summary['target_achievement']

        print(f"✅ Optimization completed in {summary['optimization_time_minutes']:.1f} minutes")
        print(f"✅ Final R² Score: {test_r2:.4f}")

        print(f"\n🎯 Target Achievement:")
        if targets['professor_target_0_86']:
            print(f"   ✅ Professor's Target (R² > 0.86): ACHIEVED")
        else:
            print(f"   ❌ Professor's Target (R² > 0.86): NOT ACHIEVED")

        if targets['thesis_target_0_87']:
            print(f"   ✅ Thesis Target (R² > 0.87): ACHIEVED")
        else:
            print(f"   ⚠️  Thesis Target (R² > 0.87): NOT ACHIEVED")

        if targets['beat_linear_regression']:
            print(f"   ✅ Beat Linear Regression: ACHIEVED")
        else:
            print(f"   ❌ Beat Linear Regression: NOT ACHIEVED")

        comp = summary['baseline_comparison']
        print(f"\n📊 Performance Improvements:")
        print(f"   vs Linear Regression: {comp['improvement_vs_linear']:+.4f}")
        print(f"   vs XGBoost Baseline: {comp['improvement_vs_baseline']:+.4f}")

        overfitting = summary['overfitting_analysis']
        print(f"\n🔍 Overfitting Control:")
        print(f"   Training R²: {overfitting['train_r2']:.4f}")
        print(f"   Test R²: {overfitting['test_r2']:.4f}")
        print(f"   Gap: {overfitting['overfitting_gap']:.4f}")

        if overfitting['overfitting_gap'] < 0.05:
            print(f"   ✅ Excellent generalization!")
        elif overfitting['overfitting_gap'] < 0.10:
            print(f"   ⚠️  Moderate overfitting")
        else:
            print(f"   🚨 Significant overfitting remains")

        print(f"\n🔄 Next Steps:")
        if test_r2 > 0.86:
            print(f"   ✅ Ready for Phase 4: Explainable AI (SHAP & LIME)")
        else:
            print(f"   ⚠️  Consider additional optimization strategies")
        print(f"   📊 Dashboard development with optimized model")


def main():
    """Main execution function for XGBoost optimization."""
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/models', exist_ok=True)

    optimizer = XGBoostOptimizer()

    if not optimizer.load_baseline_results():
        print("❌ Cannot proceed without baseline results!")
        return

    df = optimizer.load_and_prepare_data()
    X, y, label_encoders = optimizer.prepare_features_for_optimization(df)

    X_train, X_val, X_test, y_train, y_val, y_test = optimizer.split_data_consistent(X, y)

    optimizer.X_test = X_test
    optimizer.y_test = y_test

    optimization_time = optimizer.optimize_xgboost(X_train, y_train, X_val, y_val)

    test_metrics = optimizer.evaluate_optimized_model(X_train, y_train, X_val, y_val, X_test, y_test)

    optimizer.compare_with_all_baselines()

    feature_importance = optimizer.analyze_optimized_feature_importance()

    summary = optimizer.save_optimized_model(optimization_time)

    print("\n" + "=" * 60)
    print("PHASE 3B: XGBOOST OPTIMIZATION COMPLETED")
    print("=" * 60)
    print("✅ Hyperparameter optimization completed")
    print("✅ Performance evaluation and comparison completed")
    print("✅ Overfitting analysis completed")
    print("✅ Optimized model saved and ready for XAI")

    final_r2 = test_metrics['r2_score']
    if final_r2 > 0.86:
        print(f"\n🎉 SUCCESS: Achieved professor's target R² = {final_r2:.4f} > 0.86")
    else:
        print(f"\n⚠️  Target missed: R² = {final_r2:.4f} < 0.86")

    print(f"\n📅 Timeline Status:")
    print(f"   ✅ September 29: XGBoost optimization completed")
    print(f"   🔄 Next: Phase 4 - Explainable AI implementation")


if __name__ == "__main__":
    main()