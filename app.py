"""
Healthcare Cost Predictor - Streamlit Dashboard
Bachelor Thesis: XGBoost with Explainable AI
Student: Ammar Pavel Zamora Siregar (1202224044)
Universitas Telkom, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Healthcare Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Feature columns (must match training data order)
FEATURE_COLS = [
    'age', 'bmi', 'children', 'sex', 'smoker', 'region',
    'high_risk', 'smoker_bmi_interaction', 'smoker_age_interaction',
    'cost_complexity_score', 'high_risk_age_interaction',
    'bmi_category', 'age_group', 'family_size'
]

@st.cache_resource
def load_model():
    """Load the trained ensemble model"""
    try:
        with open('results/models/final_best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_training_data():
    """Load training data for SHAP/LIME background"""
    try:
        df = pd.read_csv('data/processed/insurance_enhanced_processed.csv')
        return df
    except Exception as e:
        st.error(f"Error loading training data: {e}")
        return None

@st.cache_data
def prepare_encoded_training_data(_training_data):
    """
    Encode training data EXACTLY as it was encoded during model training.
    Uses pd.Categorical().codes which creates alphabetical encoding.
    """
    X = _training_data[FEATURE_COLS].copy()

    # Encode categorical columns using pd.Categorical().codes (alphabetical order)
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_cols:
        X[col] = pd.Categorical(X[col]).codes

    # Convert all to float
    X = X.astype(float)

    return X

def categorize_bmi(bmi):
    """Categorize BMI according to WHO standards"""
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25.0:
        return 'Normal'
    elif bmi < 30.0:
        return 'Overweight'
    else:
        return 'Obese'

def categorize_age(age):
    """Categorize age into medical groups"""
    if age < 30:
        return '18-29'
    elif age < 40:
        return '30-39'
    elif age < 50:
        return '40-49'
    else:
        return '50-64'

def engineer_features(age, sex, bmi, children, smoker, region):
    """
    Engineer enhanced features from raw patient input
    Must exactly match training preprocessing
    CRITICAL: Training data uses lowercase + pd.Categorical().codes encoding
    """
    # Convert inputs to lowercase to match training data
    sex_lower = sex.lower()
    smoker_lower = smoker.lower()
    region_lower = region.lower()

    # BMI and Age categories
    bmi_category = categorize_bmi(bmi)
    age_group = categorize_age(age)

    # Binary conversions (matching training data values)
    smoker_binary = 1 if smoker_lower == 'yes' else 0

    # Categorical encoding using pd.Categorical().codes logic
    # Training data: female=0, male=1 (alphabetical order)
    sex_encoded = 0 if sex_lower == 'female' else 1

    # Training data: northeast=0, northwest=1, southeast=2, southwest=3 (alphabetical)
    region_map_lower = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
    region_encoded = region_map_lower.get(region_lower, 0)

    # BMI category encoding (alphabetical order from pd.Categorical)
    # Normal=0, Obese=1, Overweight=2, Underweight=3
    bmi_cat_map = {'Normal': 0, 'Obese': 1, 'Overweight': 2, 'Underweight': 3}
    bmi_cat_encoded = bmi_cat_map.get(bmi_category, 0)

    # Age group encoding (alphabetical order from pd.Categorical)
    # 18-29=0, 30-39=1, 40-49=2, 50-64=3
    age_grp_map = {'18-29': 0, '30-39': 1, '40-49': 2, '50-64': 3}
    age_grp_encoded = age_grp_map.get(age_group, 0)

    # Compound features
    high_risk = 1 if (smoker_binary == 1 and bmi >= 30) else 0
    smoker_bmi_interaction = smoker_binary * bmi
    smoker_age_interaction = smoker_binary * age
    high_risk_age_interaction = high_risk * age
    family_size = children + 1

    # Cost complexity score (simplified)
    cost_complexity_score = (
        (smoker_binary * 5) +
        (1 if bmi >= 30 else 0) * 3 +
        (age / 64) * 2 +
        (children * 0.5)
    )

    # Create feature dictionary in exact order
    features = {
        'age': float(age),
        'bmi': float(bmi),
        'children': float(children),
        'sex': float(sex_encoded),
        'smoker': float(smoker_binary),
        'region': float(region_encoded),
        'high_risk': float(high_risk),
        'smoker_bmi_interaction': float(smoker_bmi_interaction),
        'smoker_age_interaction': float(smoker_age_interaction),
        'cost_complexity_score': float(cost_complexity_score),
        'high_risk_age_interaction': float(high_risk_age_interaction),
        'bmi_category': float(bmi_cat_encoded),
        'age_group': float(age_grp_encoded),
        'family_size': float(family_size)
    }

    return pd.DataFrame([features])[FEATURE_COLS]

def predict_cost(model, patient_features):
    """Predict cost with confidence interval"""
    prediction = model.predict(patient_features)[0]

    # Estimate confidence interval using base estimators if available
    try:
        if hasattr(model, 'estimators_'):
            # Get predictions from all base estimators
            base_predictions = []
            for estimator in model.estimators_:
                base_pred = estimator.predict(patient_features)[0]
                base_predictions.append(base_pred)

            std = np.std(base_predictions)
            ci_lower = prediction - 1.96 * std  # 95% CI
            ci_upper = prediction + 1.96 * std
        else:
            # Fallback: assume 15% margin
            std = prediction * 0.15
            ci_lower = prediction - 1.96 * std
            ci_upper = prediction + 1.96 * std
    except:
        # Fallback
        std = prediction * 0.15
        ci_lower = prediction - 1.96 * std
        ci_upper = prediction + 1.96 * std

    return prediction, ci_lower, ci_upper

def get_risk_category(cost, smoker, bmi):
    """Categorize patient risk level"""
    if smoker == 'Yes' and bmi >= 30:
        return "🔴 High Risk", "You are in the high-risk category (smoker + obese)"
    elif smoker == 'Yes' or bmi >= 30:
        return "🟠 Medium Risk", "You have elevated risk factors"
    else:
        return "🟢 Low Risk", "You are in the low-risk category"

def main():
    # Header
    st.title("🏥 Healthcare Cost Prediction System")
    st.markdown("### Interpretable AI for Patient Empowerment")
    st.markdown("*Powered by XGBoost + SHAP + LIME | Universitas Telkom 2025*")
    st.divider()

    # Load model and data
    model = load_model()
    training_data = load_training_data()

    if model is None or training_data is None:
        st.error("⚠️ Failed to load model or training data. Please check file paths.")
        st.stop()

    # Sidebar - Patient Input Form
    with st.sidebar:
        st.header("📋 Patient Information")
        st.markdown("Enter your health information below:")

        # Personal information
        age = st.slider("Age", min_value=18, max_value=64, value=30,
                       help="Your current age (18-64 years)")

        sex = st.radio("Sex", options=['Male', 'Female'], horizontal=True)

        bmi = st.number_input("BMI (Body Mass Index)", min_value=15.0, max_value=55.0,
                             value=25.0, step=0.1,
                             help="Your BMI = weight(kg) / height(m)²")

        # Display BMI category
        bmi_cat = categorize_bmi(bmi)
        bmi_color = {'Underweight': '🔵', 'Normal': '🟢', 'Overweight': '🟠', 'Obese': '🔴'}
        st.caption(f"BMI Category: {bmi_color.get(bmi_cat, '⚪')} {bmi_cat}")

        children = st.selectbox("Number of Children", options=[0, 1, 2, 3, 4, 5],
                               index=0,
                               help="Number of dependents covered by insurance")

        smoker = st.radio("Smoking Status", options=['No', 'Yes'], horizontal=True,
                         help="Do you currently smoke?")

        region = st.selectbox("Region",
                            options=['Northeast', 'Northwest', 'Southeast', 'Southwest'],
                            help="Your residential region in the US")

        st.divider()
        predict_button = st.button("🔮 Predict Healthcare Cost", type="primary", use_container_width=True)

    # Main content area
    if predict_button:
        # Engineer features
        patient_features = engineer_features(age, sex, bmi, children, smoker, region)

        # Make prediction
        prediction, ci_lower, ci_upper = predict_cost(model, patient_features)

        # Get risk category
        risk_category, risk_message = get_risk_category(prediction, smoker, bmi)

        # Display prediction
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.markdown(f"""
                <div class="prediction-box">
                    <h3>Estimated Annual Healthcare Cost</h3>
                    <h1 style="font-size: 3rem; margin: 1rem 0;">${prediction:,.2f}</h1>
                    <p style="font-size: 0.9rem; opacity: 0.9;">
                        95% Confidence Interval: ${ci_lower:,.2f} - ${ci_upper:,.2f}
                    </p>
                </div>
            """, unsafe_allow_html=True)

        # Risk category and comparison
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Risk Category", risk_category)
            st.caption(risk_message)

        with col2:
            population_avg = 13270.42  # From EDA
            diff = prediction - population_avg
            diff_pct = (diff / population_avg) * 100
            st.metric("vs Population Average",
                     f"${abs(diff):,.0f}",
                     f"{diff_pct:+.1f}%",
                     delta_color="inverse")

        with col3:
            if smoker == 'Yes':
                smoker_avg = 32050.23
                st.metric("vs Smoker Average",
                         f"${abs(prediction - smoker_avg):,.0f}",
                         f"{((prediction - smoker_avg)/smoker_avg)*100:+.1f}%",
                         delta_color="inverse")
            else:
                nonsmoker_avg = 8434.27
                st.metric("vs Non-Smoker Average",
                         f"${abs(prediction - nonsmoker_avg):,.0f}",
                         f"{((prediction - nonsmoker_avg)/nonsmoker_avg)*100:+.1f}%",
                         delta_color="inverse")

        st.divider()

        # Tabs for different explanations
        tab1, tab2, tab3 = st.tabs(["📊 Cost Breakdown (LIME)", "🔍 Feature Impact (SHAP)", "💡 What-If Scenarios"])

        with tab1:
            st.subheader("LIME Local Explanation")
            st.markdown("**What's driving YOUR specific cost estimate?**")

            with st.spinner("Generating LIME explanation (~8 seconds)..."):
                try:
                    # Prepare training data for LIME (properly encoded)
                    X_train_encoded = prepare_encoded_training_data(training_data)

                    # Initialize LIME explainer
                    lime_explainer = LimeTabularExplainer(
                        training_data=X_train_encoded.values,
                        feature_names=FEATURE_COLS,
                        mode='regression',
                        discretize_continuous=True,
                        random_state=42
                    )

                    # Generate explanation
                    lime_exp = lime_explainer.explain_instance(
                        data_row=patient_features.values[0],
                        predict_fn=model.predict,
                        num_features=10,
                        num_samples=5000
                    )

                    # Display LIME plot
                    fig = lime_exp.as_pyplot_figure()
                    fig.set_size_inches(10, 6)
                    st.pyplot(fig)
                    plt.close()

                    # Extract top features
                    lime_values = lime_exp.as_list()

                    st.markdown("#### 💰 Top Cost Drivers/Reducers:")

                    drivers = [x for x in lime_values if x[1] > 0]
                    reducers = [x for x in lime_values if x[1] < 0]

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**📈 Cost Drivers (Increase)**")
                        for feature, value in drivers[:3]:
                            st.markdown(f"- {feature}: **+${abs(value):,.0f}**")

                    with col2:
                        st.markdown("**📉 Cost Reducers (Decrease)**")
                        for feature, value in reducers[:3]:
                            st.markdown(f"- {feature}: **-${abs(value):,.0f}**")

                except Exception as e:
                    st.error(f"Error generating LIME explanation: {e}")

        with tab2:
            st.subheader("SHAP Feature Impact Analysis")
            st.markdown("**How do your features compare globally?**")

            with st.spinner("Generating SHAP explanation..."):
                try:
                    # Prepare data (properly encoded)
                    X_train_encoded = prepare_encoded_training_data(training_data)
                    background = shap.sample(X_train_encoded, 100, random_state=42)

                    # Initialize SHAP explainer
                    shap_explainer = shap.Explainer(model.predict, background)

                    # Calculate SHAP values
                    shap_values = shap_explainer(patient_features)

                    # Waterfall plot
                    st.markdown("#### 🌊 SHAP Waterfall Plot")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.waterfall_plot(shap_values[0], show=False)
                    st.pyplot(fig)
                    plt.close()

                    # Feature importance table
                    st.markdown("#### 📋 Feature Impact Summary")

                    feature_impacts = []
                    for i, feature in enumerate(FEATURE_COLS):
                        impact = shap_values.values[0][i]
                        feature_impacts.append({
                            'Feature': feature,
                            'SHAP Value': f"${impact:,.2f}",
                            'Impact': 'Increase' if impact > 0 else 'Decrease'
                        })

                    impact_df = pd.DataFrame(feature_impacts)
                    impact_df = impact_df.sort_values('SHAP Value', key=lambda x: x.str.replace('$', '').str.replace(',', '').astype(float).abs(), ascending=False)

                    st.dataframe(impact_df.head(10), use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Error generating SHAP explanation: {e}")

        with tab3:
            st.subheader("💡 What-If Scenario Analysis")
            st.markdown("**See how lifestyle changes could affect your costs**")

            # Smoking cessation scenario
            if smoker == 'Yes':
                st.markdown("#### 🚭 Smoking Cessation Impact")

                # Calculate non-smoker scenario
                modified_features = engineer_features(age, sex, bmi, children, 'No', region)
                modified_pred, _, _ = predict_cost(model, modified_features)

                savings = prediction - modified_pred
                savings_pct = (savings / prediction) * 100

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Current Cost (Smoker)", f"${prediction:,.2f}")

                with col2:
                    st.metric("If You Quit Smoking",
                             f"${modified_pred:,.2f}",
                             f"-${savings:,.2f} ({savings_pct:.1f}% reduction)",
                             delta_color="inverse")

                st.success(f"🎯 **Potential Annual Savings: ${savings:,.2f}**")
                st.info("💡 Quitting smoking is the #1 most impactful lifestyle change you can make to reduce healthcare costs!")

            # Weight management scenario
            st.markdown("#### ⚖️ Weight Management Impact")

            target_bmi = st.slider("Target BMI", min_value=18.5, max_value=float(bmi), value=25.0, step=0.5)

            if target_bmi < bmi:
                weight_features = engineer_features(age, sex, target_bmi, children, smoker, region)
                weight_pred, _, _ = predict_cost(model, weight_features)

                weight_savings = prediction - weight_pred
                weight_savings_pct = (weight_savings / prediction) * 100

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Current Cost", f"${prediction:,.2f}")

                with col2:
                    st.metric(f"At BMI {target_bmi}",
                             f"${weight_pred:,.2f}",
                             f"-${weight_savings:,.2f} ({weight_savings_pct:.1f}% reduction)",
                             delta_color="inverse")

                st.success(f"🎯 **Potential Annual Savings: ${weight_savings:,.2f}**")

            # Combined scenario
            if smoker == 'Yes' and bmi > 25.0:
                st.markdown("#### 🎯 Combined Lifestyle Change Impact")

                combined_features = engineer_features(age, sex, 25.0, children, 'No', region)
                combined_pred, _, _ = predict_cost(model, combined_features)

                combined_savings = prediction - combined_pred
                combined_savings_pct = (combined_savings / prediction) * 100

                st.metric("Total Potential Savings (Quit Smoking + Healthy Weight)",
                         f"${combined_savings:,.2f}",
                         f"{combined_savings_pct:.1f}% reduction",
                         delta_color="inverse")

                st.balloons()
                st.success("🌟 **This is your maximum savings potential through lifestyle interventions!**")

    else:
        # Welcome screen
        st.info("👈 **Enter your information in the sidebar and click 'Predict' to get started!**")

        st.markdown("### 🎯 About This Tool")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **What This Tool Does:**
            - Predicts your annual healthcare costs
            - Explains what drives your costs (SHAP & LIME)
            - Shows savings from lifestyle changes
            - Provides personalized recommendations

            **How It Works:**
            - XGBoost ensemble model (R² = 0.877)
            - Trained on 1,338 patient records
            - Dual Explainable AI framework
            """)

        with col2:
            st.markdown("""
            **Key Findings from Our Research:**
            - 🚬 Smoking increases costs by **280%**
            - ⚖️ Obesity + Smoking increases costs by **370%**
            - 💰 Average potential savings: **$8,000 - $45,200**
            - 🎯 100% of high-cost cases are smokers

            **Your Privacy:**
            - No data is stored or shared
            - All calculations run locally
            - Secure and confidential
            """)

    # Footer
    st.divider()
    st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            <p>
                <strong>Bachelor Thesis Project</strong><br>
                Ammar Pavel Zamora Siregar (1202224044)<br>
                School of Informatics, Universitas Telkom, 2025<br>
                <em>Supervised by: Indra Aulia, S.TI., M.Kom. & Nurul Ilmi, S.Kom, M.T</em>
            </p>
            <p style="margin-top: 0.5rem;">
                ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only.
                Predictions are estimates and should not replace professional medical or financial advice.
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
