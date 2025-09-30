"""
Phase 3D: TARGETED XGBOOST OPTIMIZATION - THESIS TARGET ACHIEVEMENT
============================================================================
Strategy: Focus on proven high-value features + systematic hyperparameter tuning
Target: R² ≥ 0.87 (MUST ACHIEVE!)

Analysis of Previous Results:
- Enhanced Linear Regression: R² = 0.8637 (baseline to beat)
- Enhanced XGBoost: R² = 0.8618 (close, but overfitting gap = 0.1975)
- Advanced XGBoost: R² = 0.8528 (feature bloat hurt performance)

Strategy:
1. Use ONLY proven high-value features (avoid feature bloat)
2. Systematic hyperparameter optimization (Bayesian if available)
3. Focus on reducing overfitting gap
4. Target R² ≥ 0.87 with aggressive optimization
============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import json
import pickle
import warnings
import time
from datetime import datetime
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import advanced optimization tools
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True
    print("✅ Bayesian optimization available (skopt)")
except ImportError:
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform, randint
    BAYESIAN_AVAILABLE = False
    print("⚠️  Bayesian optimization not available, using RandomizedSearchCV")

def create_results_directories():
    """Create necessary directories for saving results"""
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    print("✅ Results directories ready")

def load_enhanced_data():
    """Load the enhanced preprocessed dataset"""
    print("Loading enhanced preprocessed data...")
    try:
        df = pd.read_csv('data/processed/insurance_enhanced_processed.csv')
        print(f"✅ Enhanced dataset loaded: {len(df)} records, {len(df.columns)} features")

        # Verify data quality
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        print(f"✅ Data quality: {missing_pct:.2f}% missing values")

        return df
    except FileNotFoundError:
        print("❌ Enhanced data not found. Please run 00_enhanced_data_preprocessing.py first.")
        raise

def select_proven_features(df):
    """
    Select ONLY the proven high-value features that demonstrated strong predictive power
    Strategy: Avoid feature bloat, focus on features with highest correlation/importance
    """
    print("\n" + "="*60)
    print("PROVEN FEATURE SELECTION - AVOIDING FEATURE BLOAT")
    print("="*60)

    # Core original features (always include)
    core_features = ['age', 'bmi', 'children', 'sex', 'smoker', 'region']

    # Proven high-value enhanced features (from previous analysis)
    proven_enhanced = [
        'high_risk',                    # r=0.815, Rank 1 importance
        'smoker_bmi_interaction',       # r=0.845, Rank 2 importance
        'smoker_age_interaction',       # r=0.789, Rank 7 importance
        'cost_complexity_score',        # r=0.745, Rank 5 importance
        'high_risk_age_interaction',    # Rank 4 importance
        'bmi_category',                 # Medical domain knowledge
        'age_group',                    # Age stratification
        'family_size'                   # Simple but effective
    ]

    # Select features that exist in the dataset
    available_features = []
    for feature in core_features + proven_enhanced:
        if feature in df.columns:
            available_features.append(feature)
        else:
            print(f"⚠️  Feature not found: {feature}")

    # Create the focused dataset
    feature_df = df[available_features + ['charges']].copy()

    print(f"🎯 Focused Feature Selection:")
    print(f"   Core original features: {len(core_features)}")
    print(f"   Proven enhanced features: {len(proven_enhanced)}")
    print(f"   Total selected features: {len(available_features)}")
    print(f"   Avoided feature bloat: {len(df.columns) - len(available_features) - 1} features excluded")

    # Encode categorical variables for correlation calculation
    feature_df_encoded = feature_df.copy()
    categorical_columns = feature_df_encoded.select_dtypes(include=['object']).columns

    print(f"\n🔧 Encoding {len(categorical_columns)} categorical features for correlation analysis:")
    for col in categorical_columns:
        if col != 'charges':  # Skip target variable
            le = LabelEncoder()
            feature_df_encoded[col] = le.fit_transform(feature_df_encoded[col])
            unique_values = feature_df[col].nunique()
            print(f"   ✅ {col}: {unique_values} categories")

    # Show feature correlations with target
    correlations = feature_df_encoded.corr()['charges'].abs().sort_values(ascending=False)
    print(f"\n📊 Feature correlations with charges:")
    for feature, corr in correlations.head(10).items():
        if feature != 'charges':
            print(f"   {feature}: r={corr:.3f}")

    return feature_df

def prepare_xgboost_data(df):
    """Prepare data specifically for XGBoost training"""
    print("\n" + "="*60)
    print("XGBOOST DATA PREPARATION")
    print("="*60)

    # Separate features and target
    X = df.drop('charges', axis=1)
    y = df['charges']

    # Handle categorical variables for XGBoost
    categorical_columns = X.select_dtypes(include=['object']).columns

    print(f"🔧 Encoding categorical features:")
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        unique_values = df[col].nunique()
        print(f"   ✅ {col}: {unique_values} categories")

    # Verify data quality
    print(f"\n✅ XGBoost Data Quality Check:")
    print(f"   Missing values: {X.isnull().sum().sum()}")
    print(f"   Infinite values: {np.isinf(X).sum().sum()}")
    print(f"   Data types: {X.dtypes.value_counts().to_dict()}")

    print(f"\n📊 Final Feature Set for Optimization:")
    print(f"   Total features: {len(X.columns)}")
    print(f"   Feature names: {list(X.columns)}")
    print(f"   Target range: ${y.min():,.0f} - ${y.max():,.0f}")

    return X, y

def create_data_splits(X, y, random_state=42):
    """Create consistent train/val/test splits"""
    print("\n" + "="*60)
    print("DATA SPLITTING (CONSISTENT WITH PREVIOUS PHASES)")
    print("="*60)

    # First split: train (70%) vs temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=None
    )

    # Second split: val (15%) vs test (15%) from temp (30%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=None
    )

    print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    # Verify target distribution consistency
    print(f"\n📊 Target Distribution Consistency:")
    print(f"   Train mean: ${y_train.mean():,.0f}, std: ${y_train.std():,.0f}")
    print(f"   Val mean: ${y_val.mean():,.0f}, std: ${y_val.std():,.0f}")
    print(f"   Test mean: ${y_test.mean():,.0f}, std: ${y_test.std():,.0f}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def create_optimization_space():
    """
    Create hyperparameter search space focused on achieving R² ≥ 0.87
    Strategy: Aggressive search ranges to find optimal parameters
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION SPACE - AGGRESSIVE SEARCH")
    print("="*60)

    if BAYESIAN_AVAILABLE:
        # Bayesian optimization space (more efficient)
        search_space = {
            'n_estimators': Integer(200, 2000),           # More trees for better performance
            'max_depth': Integer(3, 12),                   # Balance depth vs overfitting
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),  # Wide range
            'subsample': Real(0.6, 1.0),                   # Prevent overfitting
            'colsample_bytree': Real(0.6, 1.0),           # Feature sampling
            'reg_alpha': Real(0.001, 10.0, prior='log-uniform'),    # L1 regularization
            'reg_lambda': Real(0.001, 10.0, prior='log-uniform'),   # L2 regularization
            'min_child_weight': Integer(1, 20),           # Prevent overfitting
            'gamma': Real(0.0, 5.0),                      # Minimum split loss
        }
        print("🚀 Using Bayesian optimization (more efficient)")
    else:
        # RandomizedSearchCV space
        search_space = {
            'n_estimators': randint(200, 2000),
            'max_depth': randint(3, 12),
            'learning_rate': uniform(0.01, 0.29),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': uniform(0.001, 9.999),
            'reg_lambda': uniform(0.001, 9.999),
            'min_child_weight': randint(1, 20),
            'gamma': uniform(0.0, 5.0),
        }
        print("⚠️  Using RandomizedSearchCV (fallback)")

    print(f"📊 Search space configured:")
    for param, space in search_space.items():
        print(f"   {param}: {space}")

    return search_space

def perform_aggressive_optimization(X_train, y_train, search_space, cv_folds=5):
    """
    Perform aggressive hyperparameter optimization to achieve R² ≥ 0.87
    """
    print("\n" + "="*60)
    print("AGGRESSIVE HYPERPARAMETER OPTIMIZATION")
    print("="*60)

    # Base XGBoost model
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        tree_method='hist'  # Faster training
    )

    # Configure optimization
    if BAYESIAN_AVAILABLE:
        optimizer = BayesSearchCV(
            estimator=xgb_model,
            search_spaces=search_space,
            n_iter=100,                    # More iterations for better results
            cv=cv_folds,
            scoring='r2',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        print(f"🚀 Starting Bayesian optimization:")
        print(f"   Iterations: 100")
        print(f"   CV folds: {cv_folds}")
        print(f"   Total fits: {100 * cv_folds}")
    else:
        optimizer = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=search_space,
            n_iter=150,                    # More iterations for better results
            cv=cv_folds,
            scoring='r2',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        print(f"🔄 Starting RandomizedSearchCV:")
        print(f"   Iterations: 150")
        print(f"   CV folds: {cv_folds}")
        print(f"   Total fits: {150 * cv_folds}")

    # Perform optimization
    start_time = time.time()
    optimizer.fit(X_train, y_train)
    optimization_time = time.time() - start_time

    print(f"✅ Optimization completed in {optimization_time/60:.1f} minutes")
    print(f"✅ Best CV R² Score: {optimizer.best_score_:.4f}")

    # Show best parameters
    print(f"\n🏆 Best Parameters Found:")
    for param, value in optimizer.best_params_.items():
        print(f"   {param}: {value}")

    return optimizer.best_estimator_, optimizer.best_score_

def evaluate_model_performance(model, X_train, X_val, X_test, y_train, y_val, y_test):
    """Comprehensive model evaluation"""
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)

    def calculate_metrics(y_true, y_pred, dataset_name):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return {
            'dataset': dataset_name,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred, 'Training')
    val_metrics = calculate_metrics(y_val, y_val_pred, 'Validation')
    test_metrics = calculate_metrics(y_test, y_test_pred, 'Test')

    # Display results
    print(f"📊 Optimized XGBoost Performance:")
    print(f"\nTraining Performance:")
    print(f"  R² Score: {train_metrics['r2']:.4f}")
    print(f"  RMSE: ${train_metrics['rmse']:,.2f}")
    print(f"  MAE: ${train_metrics['mae']:,.2f}")
    print(f"  MAPE: {train_metrics['mape']:.2f}%")

    print(f"\nValidation Performance:")
    print(f"  R² Score: {val_metrics['r2']:.4f}")
    print(f"  RMSE: ${val_metrics['rmse']:,.2f}")
    print(f"  MAE: ${val_metrics['mae']:,.2f}")
    print(f"  MAPE: {val_metrics['mape']:.2f}%")

    print(f"\nTest Performance:")
    print(f"  R² Score: {test_metrics['r2']:.4f}")
    print(f"  RMSE: ${test_metrics['rmse']:,.2f}")
    print(f"  MAE: ${test_metrics['mae']:,.2f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")

    # Overfitting analysis
    overfitting_gap = train_metrics['r2'] - test_metrics['r2']
    print(f"\n🔍 Model Generalization Analysis:")
    print(f"   Training R²: {train_metrics['r2']:.4f}")
    print(f"   Test R²: {test_metrics['r2']:.4f}")
    print(f"   Overfitting gap: {overfitting_gap:.4f}")

    if overfitting_gap < 0.05:
        print(f"   ✅ Excellent generalization!")
    elif overfitting_gap < 0.10:
        print(f"   ✅ Good generalization")
    else:
        print(f"   ⚠️  Some overfitting detected")

    return {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics,
        'overfitting_gap': overfitting_gap
    }

def analyze_thesis_target_achievement(test_r2, previous_results=None):
    """Analyze if we achieved the thesis target R² ≥ 0.87"""
    print("\n" + "="*60)
    print("THESIS TARGET ACHIEVEMENT ANALYSIS")
    print("="*60)

    thesis_target = 0.87
    gap_to_target = thesis_target - test_r2

    print(f"🎯 THESIS TARGET ANALYSIS:")
    print(f"   📊 Current Performance: R² = {test_r2:.4f}")
    print(f"   🏆 Thesis Target: R² ≥ {thesis_target}")
    print(f"   📈 Gap to target: {gap_to_target:.4f}")

    if test_r2 >= thesis_target:
        print(f"   🎉 ✅ THESIS TARGET ACHIEVED! R² = {test_r2:.4f} ≥ {thesis_target}")
        achievement_status = "ACHIEVED"
    else:
        print(f"   ❌ Thesis target not yet achieved")
        print(f"   📊 Performance improvement needed: {gap_to_target:.4f} ({gap_to_target/thesis_target*100:.2f}%)")
        achievement_status = "NOT_ACHIEVED"

    # Compare with previous results if provided
    if previous_results:
        print(f"\n📈 Model Evolution Comparison:")
        print(f"   Enhanced Linear Regression: R² = 0.8637")
        print(f"   Enhanced XGBoost Baseline: R² = 0.8618")
        print(f"   Advanced XGBoost (3c): R² = 0.8528")
        print(f"   Targeted XGBoost (3d): R² = {test_r2:.4f}")

        improvement_from_baseline = test_r2 - 0.8618
        improvement_from_linear = test_r2 - 0.8637

        print(f"\n📊 Improvement Analysis:")
        print(f"   vs Enhanced XGBoost: {improvement_from_baseline:+.4f}")
        print(f"   vs Linear Regression: {improvement_from_linear:+.4f}")

        if improvement_from_baseline > 0:
            print(f"   ✅ Improved over XGBoost baseline")
        if improvement_from_linear > 0:
            print(f"   ✅ Improved over Linear Regression")

    return {
        'test_r2': test_r2,
        'thesis_target': thesis_target,
        'gap_to_target': gap_to_target,
        'achievement_status': achievement_status
    }

def analyze_feature_importance(model, feature_names):
    """Analyze feature importance for the optimized model"""
    print("\n" + "="*60)
    print("OPTIMIZED MODEL FEATURE IMPORTANCE")
    print("="*60)

    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"🔍 Top {min(12, len(importance_df))} Most Important Features:")
    print("Feature                           Importance")
    print("-" * 50)
    for _, row in importance_df.head(12).iterrows():
        print(f"{row['feature']:<30} {row['importance']:.6f}")

    # Analyze feature categories
    original_features = ['age', 'bmi', 'children', 'sex', 'smoker', 'region']
    enhanced_features = [f for f in feature_names if f not in original_features]

    top_10_features = importance_df.head(10)['feature'].tolist()
    original_in_top10 = len([f for f in top_10_features if f in original_features])
    enhanced_in_top10 = len([f for f in top_10_features if f in enhanced_features])

    print(f"\n📊 Feature Category Analysis (Top 10):")
    print(f"   Original features: {original_in_top10}")
    print(f"   Enhanced features: {enhanced_in_top10}")
    print(f"   Enhancement effectiveness: {enhanced_in_top10/10*100:.0f}% of top features")

    return importance_df

def save_optimization_results(model, metrics, importance_df, target_analysis):
    """Save all optimization results"""
    print("\n" + "="*60)
    print("SAVING OPTIMIZATION RESULTS")
    print("="*60)

    # Save model
    model_path = 'results/models/xgboost_targeted_optimized.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Optimized model saved: {model_path}")

    # Prepare summary data
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'XGBoost Targeted Optimization',
        'optimization_strategy': 'Focused features + aggressive hyperparameter tuning',
        'performance': {
            'train_r2': metrics['train']['r2'],
            'validation_r2': metrics['validation']['r2'],
            'test_r2': metrics['test']['r2'],
            'test_rmse': metrics['test']['rmse'],
            'test_mae': metrics['test']['mae'],
            'test_mape': metrics['test']['mape'],
            'overfitting_gap': metrics['overfitting_gap']
        },
        'target_analysis': target_analysis,
        'feature_importance': importance_df.to_dict('records'),
        'model_parameters': model.get_params()
    }

    # Save summary
    summary_path = 'results/models/xgboost_targeted_optimized_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"✅ Optimization summary saved: {summary_path}")

    return summary

def main():
    """Main execution function"""
    print("=" * 70)
    print("PHASE 3D: TARGETED XGBOOST OPTIMIZATION - THESIS TARGET ACHIEVEMENT")
    print("=" * 70)
    print("🎯 MISSION: ACHIEVE R² ≥ 0.87 (MUST ACHIEVE!)")
    print("🚀 Strategy: Focused features + Aggressive hyperparameter optimization")
    print("=" * 70)

    # Setup
    create_results_directories()

    try:
        # Load and prepare data
        df = load_enhanced_data()
        focused_df = select_proven_features(df)
        X, y = prepare_xgboost_data(focused_df)
        X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(X, y)

        # Optimization
        search_space = create_optimization_space()
        optimized_model, best_cv_score = perform_aggressive_optimization(X_train, y_train, search_space)

        # Evaluation
        metrics = evaluate_model_performance(optimized_model, X_train, X_val, X_test, y_train, y_val, y_test)
        target_analysis = analyze_thesis_target_achievement(metrics['test']['r2'])
        importance_df = analyze_feature_importance(optimized_model, X.columns.tolist())

        # Save results
        summary = save_optimization_results(optimized_model, metrics, importance_df, target_analysis)

        # Final summary
        print("\n" + "=" * 70)
        print("TARGETED XGBOOST OPTIMIZATION - FINAL RESULTS")
        print("=" * 70)
        print(f"✅ Targeted optimization completed successfully!")
        print(f"📊 Final Test R² Score: {metrics['test']['r2']:.4f}")
        print(f"🎯 Thesis Target (≥0.87): {target_analysis['achievement_status']}")

        if target_analysis['achievement_status'] == "ACHIEVED":
            print(f"🎉 🏆 THESIS TARGET ACHIEVED! 🏆 🎉")
            print(f"📈 R² = {metrics['test']['r2']:.4f} ≥ 0.87")
            print(f"🎓 Ready for Phase 4: Explainable AI implementation")
        else:
            print(f"⚠️  Target gap remaining: {target_analysis['gap_to_target']:.4f}")
            print(f"💡 Consider additional strategies if needed")

        print(f"\n⏱️  Model Quality:")
        print(f"   Overfitting gap: {metrics['overfitting_gap']:.4f}")
        print(f"   Generalization: {'Excellent' if metrics['overfitting_gap'] < 0.05 else 'Good' if metrics['overfitting_gap'] < 0.10 else 'Fair'}")

        print(f"\n🔄 Next Steps:")
        if target_analysis['achievement_status'] == "ACHIEVED":
            print(f"   ✅ Phase 4: Implement SHAP & LIME explainability")
            print(f"   ✅ Phase 5: Dashboard development")
            print(f"   ✅ Thesis documentation with strong results")
        else:
            print(f"   🔄 Consider ensemble methods or data augmentation")
            print(f"   🔄 Review feature engineering strategies")
            print(f"   🔄 Advanced ensemble techniques")

        print("=" * 70)
        print("PHASE 3D: TARGETED OPTIMIZATION COMPLETED")
        print("=" * 70)

        return summary

    except Exception as e:
        print(f"❌ Error in targeted optimization: {str(e)}")
        raise

if __name__ == "__main__":
    start_time = time.time()
    results = main()
    total_time = time.time() - start_time
    print(f"\n⏱️  Total execution time: {total_time/60:.1f} minutes")