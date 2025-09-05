"""
Phase 1: Data Exploration and Analysis
XGBoost Explainable AI for Patient Treatment Cost Prediction

Author: Ammar Pavel Zamora Siregar (1202224044)
Date: January 2025
Objective: Comprehensive exploratory data analysis of the Insurance Cost dataset

This script performs the initial data exploration for the thesis project on 
implementing XGBoost with Explainable AI (SHAP & LIME) for patient treatment cost prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Set random seed for reproducibility
np.random.seed(42)

def load_and_inspect_data():
    """Load the dataset and perform initial inspection."""
    print("=" * 60)
    print("PHASE 1: DATA EXPLORATION AND ANALYSIS")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv('data/raw/insurance.csv')
    
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Total records: {len(df):,}")
    print(f"Features: {list(df.columns)}")
    
    print("\nFirst 10 rows:")
    print(df.head(10))
    
    print("\nDataset Information:")
    print(df.info())
    
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    return df

def analyze_missing_values(df):
    """Analyze missing values in the dataset."""
    print("\n" + "=" * 60)
    print("MISSING VALUES ANALYSIS")
    print("=" * 60)
    
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percentage
    })
    
    print("Missing Values Analysis:")
    print(missing_df)
    
    if missing_data.sum() > 0:
        print(f"\nTotal missing values: {missing_data.sum()}")
    else:
        print("\nNo missing values found!")

def analyze_target_variable(df):
    """Analyze the target variable (charges) distribution."""
    print("\n" + "=" * 60)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 60)
    
    print("Target Variable (Charges) Statistics:")
    print(df['charges'].describe())
    
    print(f"\nRange: ${df['charges'].min():,.2f} - ${df['charges'].max():,.2f}")
    print(f"IQR: ${df['charges'].quantile(0.75) - df['charges'].quantile(0.25):,.2f}")
    print(f"Skewness: {df['charges'].skew():.3f}")
    print(f"Kurtosis: {df['charges'].kurtosis():.3f}")
    
    
    # Visualize charges distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Histogram
    axes[0, 0].hist(df['charges'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Insurance Charges', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Charges ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(df['charges'].mean(), color='red', linestyle='--', label=f'Mean: ${df["charges"].mean():.0f}')
    axes[0, 0].axvline(df['charges'].median(), color='green', linestyle='--', label=f'Median: ${df["charges"].median():.0f}')
    axes[0, 0].legend()
    
    # Box plot
    axes[0, 1].boxplot(df['charges'])
    axes[0, 1].set_title('Box Plot of Insurance Charges', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Charges ($)')
    
    # Q-Q plot
    stats.probplot(df['charges'], dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normal Distribution)', fontsize=14, fontweight='bold')
    
    # Log-transformed distribution
    log_charges = np.log1p(df['charges'])
    axes[1, 1].hist(log_charges, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 1].set_title('Log-Transformed Charges Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Log(1 + Charges)')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('results/plots/01_target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Log-transformed skewness: {log_charges.skew():.3f}")

def analyze_categorical_features(df):
    """Analyze categorical features distribution and impact."""
    print("\n" + "=" * 60)
    print("CATEGORICAL FEATURES ANALYSIS")
    print("=" * 60)
    
    categorical_cols = ['sex', 'smoker', 'region']
    
    print("Categorical Features Overview:")
    for col in categorical_cols:
        print(f"\n{col.upper()}:")
        value_counts = df[col].value_counts()
        percentages = (value_counts / len(df)) * 100
        
        result_df = pd.DataFrame({
            'Count': value_counts,
            'Percentage': percentages.round(2)
        })
        print(result_df)
    
    # Visualize categorical features
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Sex distribution
    sex_counts = df['sex'].value_counts()
    axes[0, 0].pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Sex Distribution', fontsize=14, fontweight='bold')
    
    # Smoker distribution
    smoker_counts = df['smoker'].value_counts()
    axes[0, 1].pie(smoker_counts.values, labels=smoker_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Smoker Distribution', fontsize=14, fontweight='bold')
    
    # Region distribution
    region_counts = df['region'].value_counts()
    axes[1, 0].bar(region_counts.index, region_counts.values, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('Region Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Region')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Children distribution
    children_counts = df['children'].value_counts().sort_index()
    axes[1, 1].bar(children_counts.index, children_counts.values, color='orange', alpha=0.7)
    axes[1, 1].set_title('Number of Children Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Number of Children')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('results/plots/02_categorical_features.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_numerical_features(df):
    """Analyze numerical features distribution."""
    print("\n" + "=" * 60)
    print("NUMERICAL FEATURES ANALYSIS")
    print("=" * 60)
    
    numerical_cols = ['age', 'bmi', 'children']
    
    print("Numerical Features Statistics:")
    print(df[numerical_cols].describe())
    
    print("\nAdditional Statistics:")
    for col in numerical_cols:
        print(f"\n{col.upper()}:")
        print(f"  Skewness: {df[col].skew():.3f}")
        print(f"  Kurtosis: {df[col].kurtosis():.3f}")
        print(f"  Range: {df[col].max() - df[col].min():.2f}")
    
    # Visualize numerical features
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Age
    axes[0, 0].hist(df['age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Age Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Age (years)')
    axes[0, 0].set_ylabel('Frequency')
    
    # BMI
    axes[0, 1].hist(df['bmi'].dropna(), bins=25, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('BMI Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('BMI (kg/m²)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(18.5, color='red', linestyle='--', alpha=0.7, label='Underweight')
    axes[0, 1].axvline(25, color='orange', linestyle='--', alpha=0.7, label='Normal')
    axes[0, 1].axvline(30, color='red', linestyle='--', alpha=0.7, label='Obese')
    axes[0, 1].legend()
    
    # Children
    children_counts = df['children'].value_counts().sort_index()
    axes[0, 2].bar(children_counts.index, children_counts.values, color='orange', alpha=0.7)
    axes[0, 2].set_title('Children Distribution', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Number of Children')
    axes[0, 2].set_ylabel('Count')
    
    # Box plots
    numerical_data = [df['age'], df['bmi'].dropna(), df['children']]
    numerical_labels = ['Age', 'BMI', 'Children']
    
    for i, (data, label) in enumerate(zip(numerical_data, numerical_labels)):
        axes[1, i].boxplot(data)
        axes[1, i].set_title(f'{label} Box Plot', fontsize=14, fontweight='bold')
        axes[1, i].set_ylabel(label)
    
    plt.tight_layout()
    plt.savefig('results/plots/03_numerical_features.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_correlations(df):
    """Analyze feature correlations with charges."""
    print("\n" + "=" * 60)
    print("FEATURE CORRELATIONS ANALYSIS")
    print("=" * 60)
    
    # Create encoded dataset for correlation analysis
    df_encoded = df.copy()
    df_encoded['sex_encoded'] = df_encoded['sex'].map({'female': 0, 'male': 1})
    df_encoded['smoker_encoded'] = df_encoded['smoker'].map({'no': 0, 'yes': 1})
    df_encoded['region_encoded'] = pd.Categorical(df_encoded['region']).codes
    
    # Select numerical columns for correlation
    corr_cols = ['age', 'sex_encoded', 'bmi', 'children', 'smoker_encoded', 'region_encoded', 'charges']
    correlation_matrix = df_encoded[corr_cols].corr()
    
    print("Correlation Matrix:")
    print(correlation_matrix.round(3))
    
    # Correlation with charges
    charges_corr = correlation_matrix['charges'].drop('charges').abs().sort_values(ascending=False)
    print("\nAbsolute Correlation with Charges (sorted):")
    feature_names = {
        'smoker_encoded': 'smoker',
        'sex_encoded': 'sex',
        'region_encoded': 'region'
    }
    for feature, corr in charges_corr.items():
        display_name = feature_names.get(feature, feature)
        print(f"  {display_name}: {corr:.3f}")
    
    # Visualize correlation matrix
    plt.figure(figsize=(12, 10))
    labels = ['Age', 'Sex', 'BMI', 'Children', 'Smoker', 'Region', 'Charges']
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                mask=mask,
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={"shrink": .8})
    
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('results/plots/04_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_impact(df):
    """Analyze how each feature affects insurance charges."""
    print("\n" + "=" * 60)
    print("FEATURE IMPACT ON CHARGES")
    print("=" * 60)
    
    # Charges by categorical features
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Charges by sex
    sns.boxplot(data=df, x='sex', y='charges', ax=axes[0, 0])
    axes[0, 0].set_title('Charges by Sex', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Charges ($)')
    
    # Charges by smoker status
    sns.boxplot(data=df, x='smoker', y='charges', ax=axes[0, 1])
    axes[0, 1].set_title('Charges by Smoker Status', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Charges ($)')
    
    # Charges by region
    sns.boxplot(data=df, x='region', y='charges', ax=axes[1, 0])
    axes[1, 0].set_title('Charges by Region', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Charges ($)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Charges by number of children
    sns.boxplot(data=df, x='children', y='charges', ax=axes[1, 1])
    axes[1, 1].set_title('Charges by Number of Children', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Charges ($)')
    
    plt.tight_layout()
    plt.savefig('results/plots/05_feature_impact.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical analysis of categorical features impact
    print("Average Charges by Categorical Features:")
    
    categorical_features = ['sex', 'smoker', 'region']
    
    for feature in categorical_features:
        print(f"\n{feature.upper()}:")
        avg_charges = df.groupby(feature)['charges'].agg(['mean', 'median', 'count']).round(2)
        avg_charges.columns = ['Average', 'Median', 'Count']
        print(avg_charges)
        
        # Calculate percentage difference from overall mean
        overall_mean = df['charges'].mean()
        print("\nPercentage difference from overall mean:")
        for category in df[feature].unique():
            cat_mean = df[df[feature] == category]['charges'].mean()
            pct_diff = ((cat_mean - overall_mean) / overall_mean) * 100
            print(f"  {category}: {pct_diff:+.1f}%")

def analyze_feature_interactions(df):
    """Explore interactions between features."""
    print("\n" + "=" * 60)
    print("FEATURE INTERACTIONS ANALYSIS")
    print("=" * 60)
    
    # Smoking vs BMI interaction (high importance for healthcare costs)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot with smoker distinction
    smokers = df[df['smoker'] == 'yes']
    non_smokers = df[df['smoker'] == 'no']
    
    axes[0].scatter(non_smokers['bmi'], non_smokers['charges'], 
                    alpha=0.6, color='blue', label='Non-smoker')
    axes[0].scatter(smokers['bmi'], smokers['charges'], 
                    alpha=0.6, color='red', label='Smoker')
    axes[0].set_title('BMI vs Charges by Smoking Status', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('BMI (kg/m²)')
    axes[0].set_ylabel('Charges ($)')
    axes[0].legend()
    
    # Age vs charges by smoker status
    axes[1].scatter(non_smokers['age'], non_smokers['charges'], 
                    alpha=0.6, color='blue', label='Non-smoker')
    axes[1].scatter(smokers['age'], smokers['charges'], 
                    alpha=0.6, color='red', label='Smoker')
    axes[1].set_title('Age vs Charges by Smoking Status', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Age (years)')
    axes[1].set_ylabel('Charges ($)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('results/plots/06_smoking_interactions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical comparison
    print("Smoking Impact Analysis:")
    print(f"Average charges for smokers: ${smokers['charges'].mean():,.2f}")
    print(f"Average charges for non-smokers: ${non_smokers['charges'].mean():,.2f}")
    print(f"Difference: ${smokers['charges'].mean() - non_smokers['charges'].mean():,.2f}")
    print(f"Smokers pay {(smokers['charges'].mean() / non_smokers['charges'].mean() - 1) * 100:.1f}% more")

def create_bmi_categories(df):
    """Create BMI categories and analyze interaction with smoking."""
    def categorize_bmi(bmi):
        if pd.isna(bmi):
            return 'Unknown'
        elif bmi < 18.5:
            return 'Underweight'
        elif bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'
    
    df['bmi_category'] = df['bmi'].apply(categorize_bmi)
    
    # BMI category vs charges by smoking
    pivot_table = df.pivot_table(values='charges', 
                               index='bmi_category', 
                               columns='smoker', 
                               aggfunc='mean').round(2)
    
    print("\nAverage Charges by BMI Category and Smoking Status:")
    print(pivot_table)
    
    # Visualize interaction
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='bmi_category', y='charges', hue='smoker')
    plt.title('Average Charges by BMI Category and Smoking Status', fontsize=14, fontweight='bold')
    plt.xlabel('BMI Category')
    plt.ylabel('Average Charges ($)')
    plt.xticks(rotation=45)
    plt.legend(title='Smoker')
    plt.tight_layout()
    plt.savefig('results/plots/07_bmi_smoking_interaction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def analyze_outliers(df):
    """Identify and analyze outliers."""
    print("\n" + "=" * 60)
    print("OUTLIER ANALYSIS")
    print("=" * 60)
    
    def find_outliers_iqr(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    
    # Find outliers in charges
    charges_outliers = find_outliers_iqr(df, 'charges')
    bmi_outliers = find_outliers_iqr(df.dropna(), 'bmi')
    age_outliers = find_outliers_iqr(df, 'age')
    
    print("Outlier Analysis:")
    print(f"Charges outliers: {len(charges_outliers)} ({len(charges_outliers)/len(df)*100:.1f}%)")
    print(f"BMI outliers: {len(bmi_outliers)} ({len(bmi_outliers)/len(df)*100:.1f}%)")
    print(f"Age outliers: {len(age_outliers)} ({len(age_outliers)/len(df)*100:.1f}%)")
    
    if len(charges_outliers) > 0:
        print("\nTop 5 highest charges (potential outliers):")
        print(charges_outliers.nlargest(5, 'charges')[['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']])
    
    # Analyze high-cost cases
    high_cost_threshold = df['charges'].quantile(0.95)  # Top 5%
    high_cost_cases = df[df['charges'] >= high_cost_threshold]
    
    print(f"\nHigh-Cost Cases Analysis (Top 5%, threshold: ${high_cost_threshold:,.2f}):")
    print(f"Number of high-cost cases: {len(high_cost_cases)}")
    
    # Smoking status in high-cost cases
    smoker_in_high_cost = (high_cost_cases['smoker'] == 'yes').sum()
    print(f"Smokers in high-cost cases: {smoker_in_high_cost}/{len(high_cost_cases)} ({smoker_in_high_cost/len(high_cost_cases)*100:.1f}%)")
    
    print(f"Overall smoking rate: {(df['smoker'] == 'yes').sum()/len(df)*100:.1f}%")

def feature_engineering(df):
    """Create additional features for modeling."""
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    df_processed = df.copy()
    
    # Create age groups
    df_processed['age_group'] = pd.cut(df_processed['age'], 
                                     bins=[18, 30, 40, 50, 65], 
                                     labels=['18-29', '30-39', '40-49', '50-64'])
    
    # Create high-risk indicator
    df_processed['high_risk'] = ((df_processed['smoker'] == 'yes') & 
                               (df_processed['bmi'] > 30)).astype(int)
    
    # Family size
    df_processed['family_size'] = df_processed['children'] + 1
    
    # Log charges for modeling
    df_processed['log_charges'] = np.log1p(df_processed['charges'])
    
    print("Feature Engineering Completed:")
    print("New features added: age_group, bmi_category, high_risk, family_size, log_charges")
    print(f"Final dataset shape: {df_processed.shape}")
    
    return df_processed

def save_processed_data(df_processed):
    """Save processed data for next phase."""
    os.makedirs('data/processed', exist_ok=True)
    df_processed.to_csv('data/processed/insurance_processed.csv', index=False)
    print("\nProcessed data saved to: data/processed/insurance_processed.csv")
    
    print("\nSample of processed dataset:")
    sample_cols = ['age', 'age_group', 'bmi', 'bmi_category', 'smoker', 
                   'high_risk', 'family_size', 'charges', 'log_charges']
    print(df_processed[sample_cols].head())

def print_completion_message():
    """Print completion message."""
    print("\n" + "=" * 60)
    print("EDA ANALYSIS COMPLETED")
    print("=" * 60)
    print("\nExploration complete! Review the analysis results above to identify key patterns and insights.")

def main():
    """Main execution function."""
    # Create results directory
    os.makedirs('results/plots', exist_ok=True)
    
    # Load and inspect data
    df = load_and_inspect_data()
    
    # Analyze missing values
    analyze_missing_values(df)
    
    # Analyze target variable
    analyze_target_variable(df)
    
    # Analyze categorical features
    analyze_categorical_features(df)
    
    # Analyze numerical features
    analyze_numerical_features(df)
    
    # Analyze correlations
    analyze_correlations(df)
    
    # Analyze feature impact
    analyze_feature_impact(df)
    
    # Analyze feature interactions
    analyze_feature_interactions(df)
    
    # Create BMI categories and analyze
    df = create_bmi_categories(df)
    
    # Analyze outliers
    analyze_outliers(df)
    
    # Feature engineering
    df_processed = feature_engineering(df)
    
    # Save processed data
    save_processed_data(df_processed)
    
    # Print completion message
    print_completion_message()
    
    print(f"\nCheck 'results/plots/' for saved visualizations.")

if __name__ == "__main__":
    main()