import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Disease Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    return joblib.load("disease_prediction_logistic_regression.pkl")

logistic_model = load_model()

st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fb;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #111827;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .info-card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }

    .result-card {
        background: white;
        padding: 28px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 3px 12px rgba(0,0,0,0.05);
        text-align: center;
        margin-top: 15px;
    }

    .result-label {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 8px;
    }

    .result-value {
        font-size: 30px;
        font-weight: 800;
        color: #111827;
    }

    .metric-card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .metric-title {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .metric-number {
        color: #111827;
        font-size: 28px;
        font-weight: 800;
    }

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 13px;
        padding: 30px 0 10px 0;
    }

    div.stButton > button {
        height: 52px;
        border-radius: 12px;
        font-size: 17px;
        font-weight: 700;
    }

    .stNumberInput input,
    .stSelectbox div {
        border-radius: 10px;
    }

    .sidebar-title {
        font-size: 24px;
        font-weight: 800;
        color: #111827;
    }

    .sidebar-text {
        color: #6b7280;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">🩺 Disease Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### About")

    st.markdown(
        """
        <div class="sidebar-text">
        This application uses a trained
        <b>Logistic Regression</b> machine-learning
        model to predict whether the given patient
        data is classified as Disease or No Disease.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### Model Information")

    st.write("**Algorithm:** Logistic Regression")
    st.write("**Prediction:** Disease / No Disease")
    st.write("**Probability:** Available")
    st.write("**Input Features:** 9")

    st.markdown("---")

    st.warning(
        "For educational and demonstration purposes only. "
        "This application is not a medical diagnosis."
    )

st.markdown(
    '<div class="main-title">🩺 Disease Prediction Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Health Risk Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="info-card">'
    '<b>How it works:</b> Enter the patient information below. '
    'The trained Logistic Regression model will process the '
    'information and generate a prediction with probability.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">👤 Patient Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📋 Basic Information")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1
    )

with col2:
    st.markdown("### ❤️ Health Measurements")

    blood_pressure = st.number_input(
        "Blood Pressure (mmHg)",
        min_value=1,
        max_value=250,
        value=120,
        step=1
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=1,
        max_value=500,
        value=200,
        step=1
    )

    glucose = st.number_input(
        "Glucose (mg/dL)",
        min_value=1,
        max_value=500,
        value=100,
        step=1
    )

with col3:
    st.markdown("### 🏃 Lifestyle")

    heart_rate = st.number_input(
        "Heart Rate (bpm)",
        min_value=1,
        max_value=220,
        value=72,
        step=1
    )

    smoking = st.selectbox(
        "Smoking",
        ["Yes", "No"]
    )

    exercise_level = st.selectbox(
        "Exercise Level",
        ["Low", "Moderate", "High"]
    )

new_patient = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "BMI": [bmi],
    "Blood_Pressure_mmHg": [blood_pressure],
    "Cholesterol_mg_dL": [cholesterol],
    "Glucose_mg_dL": [glucose],
    "Heart_Rate_bpm": [heart_rate],
    "Smoking": [smoking],
    "Exercise_Level": [exercise_level]
})

valid_input = (
    1 <= age <= 120
    and 10 <= bmi <= 60
    and 1 <= blood_pressure <= 250
    and 1 <= cholesterol <= 500
    and 1 <= glucose <= 500
    and 1 <= heart_rate <= 220
)

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "🔍  PREDICT DISEASE",
    type="primary",
    use_container_width=True
)

if predict_button:

    if not valid_input:

        st.error(
            "Invalid input. Please check the allowed value ranges."
        )

    else:

        prediction = logistic_model.predict(new_patient)

        probability = logistic_model.predict_proba(new_patient)

        predicted_class = prediction[0]

        classes = logistic_model.classes_

        probability_data = {
            class_name: prob
            for class_name, prob in zip(
                classes,
                probability[0]
            )
        }

        disease_probability = (
            probability_data.get("Disease", 0) * 100
        )

        no_disease_probability = (
            probability_data.get("No Disease", 0) * 100
        )

        st.markdown("---")

        st.markdown(
            '<div class="section-title">📊 Prediction Result</div>',
            unsafe_allow_html=True
        )

        if predicted_class == "Disease":

            st.error(
                "⚠️ MODEL PREDICTION: DISEASE"
            )

        else:

            st.success(
                "✅ MODEL PREDICTION: NO DISEASE"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                        Disease Probability
                    </div>
                    <div class="metric-number">
                        {disease_probability:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with result_col2:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                        No Disease Probability
                    </div>
                    <div class="metric-number">
                        {no_disease_probability:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### 📈 Probability Distribution")

        probability_df = pd.DataFrame(
            {
                "Probability (%)": [
                    disease_probability,
                    no_disease_probability
                ]
            },
            index=[
                "Disease",
                "No Disease"
            ]
        )

        st.bar_chart(
            probability_df,
            height=350
        )

        st.markdown("---")

        st.markdown("### 👤 Patient Information Used")

        display_patient = new_patient.rename(
            columns={
                "Blood_Pressure_mmHg": "Blood Pressure",
                "Cholesterol_mg_dL": "Cholesterol",
                "Glucose_mg_dL": "Glucose",
                "Heart_Rate_bpm": "Heart Rate",
                "Exercise_Level": "Exercise Level"
            }
        )

        st.dataframe(
            display_patient,
            use_container_width=True,
            hide_index=True
        )

st.markdown(
    """
    <div class="footer">
        Disease Prediction Dashboard<br>
        Logistic Regression Machine Learning Model<br><br>
        For educational and demonstration purposes only.
    </div>
    """,
    unsafe_allow_html=True
)