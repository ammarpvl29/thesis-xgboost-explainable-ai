"""
Phase 2: Baseline Linear Regression Model Implementation
XGBoost Explainable AI for Patient Treatment Cost Prediction

Author: Ammar Pavel Zamora Siregar (1202224044)
Date: January 2025
Objective: Implement baseline Linear Regression model for insurance cost prediction

This script implements Algorithm 2 from the thesis methodology - a comprehensive
baseline Linear Regression model with performance evaluation that will serve as
the comparison benchmark for XGBoost improvements.

Following the thesis proposal methodology section 3.2.1:
"Linear Regression digunakan sebagai baseline untuk menunjukkan peningkatan performa"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Set random seed for reproducibility
np.random.seed(42)

class BaselineLinearRegression:
    """
    Baseline Linear Regression implementation following Algorithm 2
    from the thesis methodology.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.performance_metrics = {}
        self.training_history = []
        
    def load_and_prepare_data(self):
        """Load processed data and prepare for modeling."""
        print("=" * 60)
        print("BASELINE LINEAR REGRESSION MODEL IMPLEMENTATION")
        print("=" * 60)
        print("Loading processed insurance dataset...")
        
        # Load processed data
        df = pd.read_csv('data/processed/insurance_processed.csv')
        print(f"Dataset loaded: {df.shape[0]} records, {df.shape[1]} features")
        
        # Handle missing values in age_group (18-year-old case) 
        # This happens because age 18 is at the boundary of our age grouping
        if df['age_group'].isnull().sum() > 0:
            df['age_group'].fillna('18-29', inplace=True)
            print(f"Fixed {df['age_group'].isnull().sum()} missing age_group values")
        
        print("\nDataset preview:")
        print(df.head())
        
        return df
    
    def feature_encoding(self, df):
        """
        Encode categorical features for linear regression.
        Following thesis methodology: proper encoding for baseline comparison.
        """
        print("\n" + "=" * 40)
        print("FEATURE ENCODING FOR LINEAR REGRESSION")
        print("=" * 40)
        
        df_encoded = df.copy()
        
        # Handle missing values first
        print("Handling missing values...")
        print(f"Missing values before cleaning:")
        missing_before = df_encoded.isnull().sum()
        print(missing_before[missing_before > 0])
        
        # Fill missing age_group (18-year-old cases should be in 18-29 group)
        if df_encoded['age_group'].isnull().sum() > 0:
            df_encoded['age_group'].fillna('18-29', inplace=True)
            print("Filled missing age_group values with '18-29'")
        
        # Fill any remaining missing BMI values with median
        if df_encoded['bmi'].isnull().sum() > 0:
            median_bmi = df_encoded['bmi'].median()
            df_encoded['bmi'].fillna(median_bmi, inplace=True)
            print(f"Filled missing BMI values with median: {median_bmi:.2f}")
        
        print(f"Missing values after cleaning:")
        missing_after = df_encoded.isnull().sum()
        print(missing_after[missing_after > 0] if missing_after.sum() > 0 else "No missing values remaining!")
        
        # Encode binary categorical features
        binary_encoders = {
            'sex': {'female': 0, 'male': 1},
            'smoker': {'no': 0, 'yes': 1}
        }
        
        for feature, mapping in binary_encoders.items():
            df_encoded[feature] = df_encoded[feature].map(mapping)
            print(f"Encoded {feature}: {mapping}")
        
        # One-hot encode multi-class categorical features
        categorical_features = ['region', 'bmi_category', 'age_group']
        
        for feature in categorical_features:
            if feature in df_encoded.columns:
                dummies = pd.get_dummies(df_encoded[feature], prefix=feature, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded.drop(feature, axis=1, inplace=True)
                print(f"One-hot encoded {feature}: {list(dummies.columns)}")
        
        # Select features for modeling (exclude target and derived features)
        exclude_features = ['charges', 'log_charges']
        feature_columns = [col for col in df_encoded.columns if col not in exclude_features]
        
        X = df_encoded[feature_columns]
        y = df_encoded['charges']  # Use original charges as target
        
        # Final check for any remaining NaN values
        if X.isnull().sum().sum() > 0:
            print("WARNING: Still have NaN values in features:")
            print(X.isnull().sum()[X.isnull().sum() > 0])
            # Fill any remaining NaN values with 0 for safety
            X = X.fillna(0)
            print("Filled remaining NaN values with 0")
        
        self.feature_names = feature_columns
        
        print(f"\nFinal feature set: {len(feature_columns)} features")
        print(f"Features: {feature_columns}")
        print(f"Target: charges (${y.min():,.0f} - ${y.max():,.0f})")
        print(f"Final X shape: {X.shape}, Final y shape: {y.shape}")
        
        return X, y
    
    def split_data(self, X, y):
        """
        Split data following thesis methodology:
        70% training, 15% validation, 15% testing
        """
        print("\n" + "=" * 40)
        print("DATA SPLITTING STRATEGY")
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
    
    def train_baseline_model(self, X_train, y_train, X_val, y_val):
        """
        Train baseline Linear Regression model following Algorithm 2.
        """
        print("\n" + "=" * 40)
        print("BASELINE MODEL TRAINING - ALGORITHM 2")
        print("=" * 40)
        
        # Create pipeline with scaling and linear regression
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('linear_reg', LinearRegression())
        ])
        
        print("Training Linear Regression with standard scaling...")
        start_time = datetime.now()
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        # Calculate training metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred, "Training")
        val_metrics = self.calculate_metrics(y_val, y_val_pred, "Validation")
        
        # Store performance metrics
        self.performance_metrics = {
            'training': train_metrics,
            'validation': val_metrics,
            'training_time': training_time
        }
        
        # Cross-validation for robustness
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='r2')
        print(f"\n5-Fold Cross-Validation R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return train_metrics, val_metrics
    
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
    
    def analyze_feature_importance(self, X_train):
        """Analyze linear regression coefficients as feature importance."""
        print("\n" + "=" * 40)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 40)
        
        # Get coefficients from the trained model
        coefficients = self.model.named_steps['linear_reg'].coef_
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        print("Top 10 Most Important Features (by absolute coefficient):")
        print(feature_importance.head(10))
        
        # Visualize feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(10)
        
        colors = ['red' if coef < 0 else 'blue' for coef in top_features['coefficient']]
        bars = plt.barh(range(len(top_features)), top_features['coefficient'], color=colors, alpha=0.7)
        
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Coefficient Value')
        plt.title('Linear Regression Feature Importance\n(Top 10 Features by Absolute Coefficient)', 
                  fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, coef) in enumerate(zip(bars, top_features['coefficient'])):
            plt.text(coef + (max(top_features['coefficient']) * 0.01), i, 
                    f'{coef:,.0f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('results/plots/08_baseline_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance
    
    def evaluate_predictions(self, X_test, y_test):
        """Final evaluation on test set."""
        print("\n" + "=" * 40)
        print("FINAL TEST SET EVALUATION")
        print("=" * 40)
        
        # Make predictions on test set
        y_test_pred = self.model.predict(X_test)
        
        # Calculate test metrics
        test_metrics = self.calculate_metrics(y_test, y_test_pred, "Test")
        
        # Store test metrics
        self.performance_metrics['test'] = test_metrics
        
        # Prediction vs Actual visualization
        self.visualize_predictions(y_test, y_test_pred)
        
        return test_metrics
    
    def visualize_predictions(self, y_true, y_pred):
        """Visualize prediction performance."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Prediction vs Actual scatter plot
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color='blue')
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                        'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Charges ($)')
        axes[0, 0].set_ylabel('Predicted Charges ($)')
        axes[0, 0].set_title('Baseline Model: Predicted vs Actual Charges')
        axes[0, 0].legend()
        
        # 2. Residuals plot
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Charges ($)')
        axes[0, 1].set_ylabel('Residuals ($)')
        axes[0, 1].set_title('Residuals Plot')
        
        # 3. Residuals distribution
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_xlabel('Residuals ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residuals Distribution')
        axes[1, 0].axvline(residuals.mean(), color='red', linestyle='--', 
                          label=f'Mean: ${residuals.mean():,.0f}')
        axes[1, 0].legend()
        
        # 4. Q-Q plot of residuals
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')
        
        plt.tight_layout()
        plt.savefig('results/plots/09_baseline_model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_model_summary(self):
        """Generate comprehensive model summary for thesis documentation."""
        print("\n" + "=" * 60)
        print("BASELINE MODEL SUMMARY - ALGORITHM 2 IMPLEMENTATION")
        print("=" * 60)
        
        summary = {
            'model_type': 'Linear Regression with StandardScaler',
            'features_used': len(self.feature_names),
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"Model Type: {summary['model_type']}")
        print(f"Features Used: {summary['features_used']}")
        print(f"Training Time: {summary['performance_metrics']['training_time']:.2f} seconds")
        
        print(f"\nPerformance Summary:")
        print(f"Training R²: {summary['performance_metrics']['training']['r2_score']:.4f}")
        print(f"Validation R²: {summary['performance_metrics']['validation']['r2_score']:.4f}")
        print(f"Test R²: {summary['performance_metrics']['test']['r2_score']:.4f}")
        
        print(f"\nTest Set RMSE: ${summary['performance_metrics']['test']['rmse']:,.2f}")
        print(f"Test Set MAE: ${summary['performance_metrics']['test']['mae']:,.2f}")
        print(f"Test Set MAPE: {summary['performance_metrics']['test']['mape']:.2f}%")
        
        # Save summary for thesis documentation
        import json
        os.makedirs('results/models', exist_ok=True)
        with open('results/models/baseline_model_summary.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json_summary = json.loads(json.dumps(summary, default=convert_numpy))
            json.dump(json_summary, f, indent=2)
        
        print(f"\nModel summary saved to: results/models/baseline_model_summary.json")
        
        return summary


def main():
    """Main execution function for baseline model implementation."""
    # Create results directory
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/models', exist_ok=True)
    
    # Initialize baseline model
    baseline = BaselineLinearRegression()
    
    # Load and prepare data
    df = baseline.load_and_prepare_data()
    
    # Feature encoding
    X, y = baseline.feature_encoding(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = baseline.split_data(X, y)
    
    # Train baseline model
    train_metrics, val_metrics = baseline.train_baseline_model(X_train, y_train, X_val, y_val)
    
    # Analyze feature importance
    feature_importance = baseline.analyze_feature_importance(X_train)
    
    # Final evaluation
    test_metrics = baseline.evaluate_predictions(X_test, y_test)
    
    # Generate model summary
    summary = baseline.generate_model_summary()
    
    print("\n" + "=" * 60)
    print("BASELINE LINEAR REGRESSION IMPLEMENTATION COMPLETED")
    print("=" * 60)
    print("✅ Algorithm 2 successfully implemented")
    print("✅ Performance metrics calculated and saved")
    print("✅ Feature importance analysis completed")
    print("✅ Baseline ready for XGBoost comparison")
    print(f"\nNext Step: Implement XGBoost model to compare with baseline R² = {test_metrics['r2_score']:.4f}")


if __name__ == "__main__":
    main()