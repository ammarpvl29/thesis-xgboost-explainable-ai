"""
PHASE 4 - STEP 1: SHAP Global Explanations Implementation
Bachelor Thesis: Patient Treatment Cost Prediction Using XGBoost with Explainable AI
Student: Ammar Pavel Zamora Siregar (1202224044)

This script implements SHAP (SHapley Additive exPlanations) for the final ensemble model
to provide global interpretability and feature importance analysis.

Outputs:
- Global SHAP feature importance
- SHAP summary plots (beeswarm, bar charts)
- SHAP waterfall plots for sample patients
- SHAP dependence plots for top features
- Enhanced feature interaction analysis
"""

import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("PHASE 4 - SHAP GLOBAL EXPLANATIONS IMPLEMENTATION")
print("="*80)
print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================================================
# 1. LOAD ENHANCED DATA AND FINAL MODEL
# ============================================================================

print("\n[1/7] Loading Enhanced Dataset and Final Model...")

# Load enhanced processed data
data_path = Path("../data/processed/insurance_enhanced_processed.csv")
df = pd.read_csv(data_path)
print(f"✓ Enhanced dataset loaded: {df.shape}")

# Define feature columns - EXACTLY as used in model training (from 04d_final_push_0.87.py)
# Core features + proven enhanced features ONLY
feature_cols = [
    # Core original features
    'age', 'bmi', 'children', 'sex', 'smoker', 'region',
    # Proven high-value enhanced features (used in training)
    'high_risk', 'smoker_bmi_interaction', 'smoker_age_interaction',
    'cost_complexity_score', 'high_risk_age_interaction',
    'bmi_category', 'age_group', 'family_size'
]

# Prepare features and target
X = df[feature_cols].copy()
y = df['charges'].copy()

# Encode ALL categorical/object columns for model compatibility
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"  Encoding categorical columns: {categorical_cols}")

for col in categorical_cols:
    X[col] = pd.Categorical(X[col]).codes

# Convert all to numeric types
X = X.astype(float)

print(f"✓ Features prepared: {X.shape[1]} features")
print(f"✓ Target variable: {y.shape[0]} records")
print(f"✓ All features converted to numeric type")

# Load final ensemble model
model_path = Path("../results/models/final_best_model.pkl")
with open(model_path, 'rb') as f:
    final_model = pickle.load(f)
print(f"✓ Final ensemble model loaded: {type(final_model).__name__}")

# ============================================================================
# 2. INITIALIZE SHAP EXPLAINER
# ============================================================================

print("\n[2/7] Initializing SHAP Explainer for Ensemble Model...")

# For StackingRegressor, we need to use model-agnostic explainer
# We'll use a sample of the data as background for KernelExplainer
# to reduce computation time while maintaining accuracy

# Use a representative sample as background (100 samples for speed)
background_sample = shap.sample(X, 100, random_state=42)
print(f"✓ Background sample created: {background_sample.shape[0]} samples")

# Initialize SHAP Explainer
# Using Explainer which auto-detects the best method for the model
explainer = shap.Explainer(final_model.predict, background_sample)
print(f"✓ SHAP Explainer initialized: {type(explainer).__name__}")

# ============================================================================
# 3. CALCULATE SHAP VALUES
# ============================================================================

print("\n[3/7] Calculating SHAP Values (this may take a few minutes)...")

# Calculate SHAP values for a sample of test data (200 samples for analysis)
# Using full dataset would be too slow for StackingRegressor
shap_sample_size = 200
shap_sample = X.sample(n=shap_sample_size, random_state=42)
print(f"✓ Computing SHAP values for {shap_sample_size} samples...")

shap_values = explainer(shap_sample)
print(f"✓ SHAP values calculated: {shap_values.values.shape}")
print(f"✓ Base value (expected prediction): ${shap_values.base_values[0]:,.2f}")

# ============================================================================
# 4. GLOBAL FEATURE IMPORTANCE ANALYSIS
# ============================================================================

print("\n[4/7] Analyzing Global Feature Importance...")

# Calculate mean absolute SHAP values for global importance
shap_importance = pd.DataFrame({
    'feature': feature_cols,
    'mean_abs_shap': np.abs(shap_values.values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

print("\n📊 TOP 10 FEATURES BY GLOBAL SHAP IMPORTANCE:")
print(shap_importance.head(10).to_string(index=False))

# Save feature importance
results_dir = Path("../results/shap")
results_dir.mkdir(parents=True, exist_ok=True)

shap_importance.to_csv(results_dir / "shap_global_feature_importance.csv", index=False)
print(f"\n✓ Feature importance saved to: {results_dir / 'shap_global_feature_importance.csv'}")

# ============================================================================
# 5. GENERATE SHAP VISUALIZATIONS
# ============================================================================

print("\n[5/7] Generating SHAP Visualizations...")

plots_dir = Path("../results/plots/shap")
plots_dir.mkdir(parents=True, exist_ok=True)

# 5.1 SHAP Summary Plot (Beeswarm)
print("\n  [5.1] Creating SHAP Summary Plot (Beeswarm)...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values.values, shap_sample, feature_names=feature_cols, show=False)
plt.title("SHAP Summary Plot - Feature Impact on Healthcare Cost Predictions",
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel("SHAP Value (Impact on Predicted Cost)", fontsize=12)
plt.tight_layout()
plt.savefig(plots_dir / "shap_summary_beeswarm.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {plots_dir / 'shap_summary_beeswarm.png'}")

# 5.2 SHAP Bar Plot (Global Importance)
print("\n  [5.2] Creating SHAP Bar Plot (Global Importance)...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values.values, shap_sample, feature_names=feature_cols,
                 plot_type="bar", show=False)
plt.title("SHAP Global Feature Importance - Mean |SHAP Value|",
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Mean |SHAP Value|", fontsize=12)
plt.tight_layout()
plt.savefig(plots_dir / "shap_bar_importance.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {plots_dir / 'shap_bar_importance.png'}")

# 5.3 SHAP Waterfall Plots for Sample Patients
print("\n  [5.3] Creating SHAP Waterfall Plots for Representative Patients...")

# Select 3 representative patients: low-cost, medium-cost, high-cost
y_sample = y.loc[shap_sample.index]
low_cost_idx = y_sample.idxmin()
med_cost_idx = y_sample.iloc[(y_sample - y_sample.median()).abs().argsort()[:1]].index[0]
high_cost_idx = y_sample.idxmax()

sample_patients = [
    (low_cost_idx, "Low Cost Patient", y.loc[low_cost_idx]),
    (med_cost_idx, "Medium Cost Patient", y.loc[med_cost_idx]),
    (high_cost_idx, "High Cost Patient", y.loc[high_cost_idx])
]

for idx, patient_type, actual_cost in sample_patients:
    sample_position = shap_sample.index.get_loc(idx)

    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap_values[sample_position], show=False)
    plt.title(f"SHAP Waterfall Plot - {patient_type}\nActual Cost: ${actual_cost:,.2f}",
              fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()

    filename = f"shap_waterfall_{patient_type.lower().replace(' ', '_')}.png"
    plt.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {plots_dir / filename}")

# 5.4 SHAP Dependence Plots for Top 4 Features
print("\n  [5.4] Creating SHAP Dependence Plots for Top Features...")

top_features = shap_importance.head(4)['feature'].tolist()

for feature in top_features:
    if feature in feature_cols:
        feature_idx = feature_cols.index(feature)

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            shap_values.values,
            shap_sample,
            feature_names=feature_cols,
            show=False
        )
        plt.title(f"SHAP Dependence Plot - {feature}\nImpact on Healthcare Cost Predictions",
                  fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()

        filename = f"shap_dependence_{feature}.png"
        plt.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {plots_dir / filename}")

# ============================================================================
# 6. ENHANCED FEATURE INTERACTION ANALYSIS
# ============================================================================

print("\n[6/7] Analyzing Enhanced Feature Interactions...")

# Focus on healthcare-critical interactions
critical_interactions = [
    ('smoker_bmi_interaction', 'Smoking-BMI Synergy'),
    ('high_risk', 'High Risk Profile'),
    ('smoker_age_interaction', 'Smoking-Age Cumulative Effect'),
    ('high_risk_age_interaction', 'High Risk Age Amplification')
]

interaction_summary = []
for feature, description in critical_interactions:
    if feature in feature_cols:
        feature_idx = feature_cols.index(feature)
        mean_impact = np.abs(shap_values.values[:, feature_idx]).mean()
        interaction_summary.append({
            'feature': feature,
            'description': description,
            'mean_abs_shap_impact': mean_impact
        })

interaction_df = pd.DataFrame(interaction_summary).sort_values('mean_abs_shap_impact', ascending=False)
print("\n📊 HEALTHCARE-CRITICAL FEATURE INTERACTIONS:")
print(interaction_df.to_string(index=False))

interaction_df.to_csv(results_dir / "shap_interaction_analysis.csv", index=False)
print(f"\n✓ Interaction analysis saved to: {results_dir / 'shap_interaction_analysis.csv'}")

# ============================================================================
# 7. SAVE COMPREHENSIVE SHAP ANALYSIS SUMMARY
# ============================================================================

print("\n[7/7] Saving Comprehensive SHAP Analysis Summary...")

shap_summary = {
    "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "phase": "Phase 4 - Step 1: SHAP Global Explanations",
    "model_type": type(final_model).__name__,
    "dataset_size": len(df),
    "shap_sample_size": shap_sample_size,
    "base_value_expected_cost": float(shap_values.base_values[0]),
    "top_5_features": shap_importance.head(5)[['feature', 'mean_abs_shap']].to_dict('records'),
    "critical_interactions": interaction_df.to_dict('records'),
    "visualizations_generated": [
        "shap_summary_beeswarm.png",
        "shap_bar_importance.png",
        "shap_waterfall_low_cost_patient.png",
        "shap_waterfall_medium_cost_patient.png",
        "shap_waterfall_high_cost_patient.png"
    ] + [f"shap_dependence_{f}.png" for f in top_features],
    "key_findings": {
        "most_important_feature": shap_importance.iloc[0]['feature'],
        "most_important_feature_impact": float(shap_importance.iloc[0]['mean_abs_shap']),
        "smoking_related_features_in_top_5": len([f for f in shap_importance.head(5)['feature']
                                                   if 'smoker' in f or 'high_risk' in f])
    }
}

with open(results_dir / "shap_analysis_summary.json", 'w') as f:
    json.dump(shap_summary, f, indent=2)

print(f"\n✓ SHAP analysis summary saved to: {results_dir / 'shap_analysis_summary.json'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ SHAP GLOBAL EXPLANATIONS IMPLEMENTATION COMPLETE")
print("="*80)
print(f"\n📁 Results Location: {results_dir}")
print(f"📊 Plots Location: {plots_dir}")
print(f"\n🔑 Key Findings:")
print(f"   • Most Important Feature: {shap_summary['key_findings']['most_important_feature']}")
print(f"   • Mean |SHAP| Impact: {shap_summary['key_findings']['most_important_feature_impact']:.4f}")
print(f"   • Base Expected Cost: ${shap_summary['base_value_expected_cost']:,.2f}")
print(f"   • Smoking-Related Features in Top 5: {shap_summary['key_findings']['smoking_related_features_in_top_5']}")
print(f"\n📈 Visualizations Generated: {len(shap_summary['visualizations_generated'])} plots")
print("\n" + "="*80)
print("READY FOR PHASE 4 - STEP 2: LIME LOCAL EXPLANATIONS")
print("="*80)
