import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    layout="centered"
)


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("heart.csv")

    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    # Remove spaces from text values
    for column in df.columns:

        if df[column].dtype == "object":

            df[column] = (
                df[column]
                .astype(str)
                .str.strip()
                .str.lower()
            )

    # --------------------------------------------------------
    # Convert THAL values
    # --------------------------------------------------------

    if "thal" in df.columns:

        thal_mapping = {
            "normal": 0,
            "fixed": 1,
            "reversible": 2,
            "fixed defect": 1,
            "reversible defect": 2,
            "unknown": 3,
            "nan": 3
        }

        df["thal"] = df["thal"].map(thal_mapping)

    # --------------------------------------------------------
    # Convert other possible categorical values
    # --------------------------------------------------------

    if "sex" in df.columns:

        df["sex"] = df["sex"].replace({
            "female": 0,
            "male": 1
        })

    if "fbs" in df.columns:

        df["fbs"] = df["fbs"].replace({
            "no": 0,
            "yes": 1
        })

    if "exang" in df.columns:

        df["exang"] = df["exang"].replace({
            "no": 0,
            "yes": 1
        })

    # --------------------------------------------------------
    # Convert all remaining columns to numeric
    # --------------------------------------------------------

    for column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove rows containing missing values
    df = df.dropna()

    # Convert everything to numeric
    df = df.astype(float)

    return df


# ============================================================
# TRAIN MACHINE LEARNING MODELS
# ============================================================

@st.cache_resource
def train_models(_df):

    # Separate features and target
    X = _df.drop("target", axis=1)
    y = _df["target"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Machine learning models
    models = {

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    }

    results = {}

    # Train each model
    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        results[name] = {
            "model": model,
            "accuracy": accuracy
        }

    return results, X.columns.tolist()


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = load_data()

except Exception as e:

    st.error(
        f"Error loading heart.csv: {e}"
    )

    st.stop()


# ============================================================
# CHECK DATASET
# ============================================================

required_columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        "Your heart.csv file is missing these columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# TRAIN MODELS
# ============================================================

try:

    all_results, feature_names = train_models(df)

except Exception as e:

    st.error(
        f"Error training the machine learning models: {e}"
    )

    st.stop()


# ============================================================
# FIND BEST MODEL
# ============================================================

best_name = max(
    all_results,
    key=lambda name: all_results[name]["accuracy"]
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Comparison")

comparison_df = pd.DataFrame({
    "Model": list(all_results.keys()),

    "Accuracy": [
        f"{all_results[name]['accuracy']:.1%}"
        for name in all_results
    ]
})

st.sidebar.dataframe(
    comparison_df,
    hide_index=True
)

st.sidebar.write(
    f"Best performer: {best_name}"
)


# ============================================================
# MODEL SELECTOR
# ============================================================

selected_model_name = st.sidebar.selectbox(
    "Choose prediction model",
    options=list(all_results.keys()),
    index=list(all_results.keys()).index(best_name)
)

model = all_results[selected_model_name]["model"]

accuracy = all_results[selected_model_name]["accuracy"]


st.sidebar.metric(
    "Selected Model Accuracy",
    f"{accuracy:.1%}"
)

st.sidebar.write(
    f"Trained on {len(df)} patient records"
)


# ============================================================
# MAIN APPLICATION
# ============================================================

st.title("Heart Disease Risk Predictor")

st.write(
    "Enter patient information below to predict heart disease risk."
)

st.info(
    "This tool is for educational purposes only and should not replace professional medical advice."
)


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.header("Patient Information")

col1, col2 = st.columns(2)


# ============================================================
# COLUMN 1
# ============================================================

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50
    )

    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x:
        "Female" if x == 0 else "Male"
    )

    cp = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic"
        ][x]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=80,
        max_value=250,
        value=120
    )

    chol = st.number_input(
        "Serum Cholesterol (mg/dl)",
        min_value=100,
        max_value=600,
        value=200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=[0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )

    restecg = st.selectbox(
        "Resting ECG Results",
        options=[0, 1, 2],
        format_func=lambda x: [
            "Normal",
            "ST-T Abnormality",
            "Left Ventricular Hypertrophy"
        ][x]
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    thalach = st.number_input(
        "Max Heart Rate Achieved",
        min_value=60,
        max_value=250,
        value=150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        options=[0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slope = st.selectbox(
        "Slope of Peak Exercise ST",
        options=[0, 1, 2],
        format_func=lambda x: [
            "Upsloping",
            "Flat",
            "Downsloping"
        ][x]
    )

    ca = st.selectbox(
        "Number of Major Vessels (0-3)",
        options=[0, 1, 2, 3]
    )

    thal = st.selectbox(
        "Thalassemia",
        options=[0, 1, 2, 3],
        format_func=lambda x: [
            "Normal",
            "Fixed Defect",
            "Reversible Defect",
            "Unknown"
        ][x]
    )


# ============================================================
# PREDICTION
# ============================================================

if st.button(
        "Predict Risk",
        type="primary"
):

    # Create input DataFrame
    input_data = pd.DataFrame(
        [[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]],
        columns=feature_names
    )

    # Make prediction
    prediction = model.predict(
        input_data
    )[0]

    probability = model.predict_proba(
        input_data
    )[0][1]


    # ========================================================
    # PREDICTION RESULTS
    # ========================================================

    st.divider()

    st.header("Prediction Results")

    st.caption(
        f"Using: {selected_model_name}"
    )


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if probability < 0.30:

        risk_level = "Low Risk"

        st.success(
            f"{risk_level} of Heart Disease "
            f"({probability:.1%} probability)"
        )

    elif probability < 0.70:

        risk_level = "Moderate Risk"

        st.warning(
            f"{risk_level} of Heart Disease "
            f"({probability:.1%} probability)"
        )

    else:

        risk_level = "High Risk"

        st.error(
            f"{risk_level} of Heart Disease "
            f"({probability:.1%} probability)"
        )


    # ========================================================
    # PROGRESS BAR
    # ========================================================

    st.subheader("Risk Level")

    st.progress(
        int(probability * 100)
    )


    # ========================================================
    # METRICS
    # ========================================================

    metric1, metric2, metric3 = st.columns(3)


    with metric1:

        st.metric(
            "Risk Probability",
            f"{probability:.1%}"
        )


    with metric2:

        st.metric(
            "Prediction",
            "Positive"
            if prediction == 1
            else "Negative"
        )


    with metric3:

        st.metric(
            "Model Accuracy",
            f"{accuracy:.1%}"
        )


    # ========================================================
    # HEALTH RECOMMENDATIONS
    # ========================================================

    st.subheader(
        "Health Recommendations"
    )


    if probability < 0.30:

        st.info(
            """
            Continue exercising regularly.

            Maintain a healthy diet.

            Schedule routine health check-ups.

            Monitor blood pressure and cholesterol.
            """
        )

    elif probability < 0.70:

        st.warning(
            """
            Increase physical activity.

            Reduce saturated fats and sodium.

            Monitor cholesterol levels.

            Consult a healthcare professional.
            """
        )

    else:

        st.error(
            """
            Seek medical advice promptly.

            Monitor blood pressure closely.

            Improve dietary habits.

            Follow prescribed treatment plans.
            """
        )


    # ========================================================
    # HEALTH ALERTS
    # ========================================================

    st.subheader(
        "Health Alerts"
    )

    alerts_found = False


    if trestbps > 140:

        st.warning(
            "High blood pressure detected."
        )

        alerts_found = True


    if chol > 240:

        st.warning(
            "High cholesterol detected."
        )

        alerts_found = True


    if exang == 1:

        st.warning(
            "Exercise-induced angina reported."
        )

        alerts_found = True


    if oldpeak > 2:

        st.warning(
            "Elevated ST depression detected."
        )

        alerts_found = True


    if not alerts_found:

        st.success(
            "No additional health alerts detected."
        )


    # ========================================================
    # TOP RISK FACTORS
    # ========================================================

    st.subheader(
        "Top Risk Factors"
    )


    if hasattr(
            model,
            "feature_importances_"
    ):

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        })

    else:

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": np.abs(
                model.coef_[0]
            )
        })


    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).head(10)


    # ========================================================
    # BAR CHART
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax.set_title(
        "Top Risk Factors"
    )

    ax.set_xlabel(
        "Importance"
    )

    ax.invert_yaxis()

    st.pyplot(fig)


    # ========================================================
    # DOWNLOAD REPORT
    # ========================================================

    st.subheader(
        "Download Report"
    )


    report = f"""
Heart Disease Risk Report
==========================

Risk Level: {risk_level}

Risk Probability: {probability:.2%}

Prediction: {"Positive" if prediction == 1 else "Negative"}

Model Used: {selected_model_name}

Model Accuracy: {accuracy:.2%}

Dataset Size: {len(df)} patients

Important:
This report is generated by an educational
machine learning application.

It should not replace professional
medical advice.
"""


    st.download_button(
        label="Download Risk Report",
        data=report,
        file_name="heart_risk_report.txt",
        mime="text/plain"
    )


# ============================================================
# SIDEBAR MODEL INFORMATION
# ============================================================

st.sidebar.divider()

st.sidebar.title(
    "Model Information"
)

st.sidebar.metric(
    "Best Model",
    best_name
)

st.sidebar.metric(
    "Best Accuracy",
    f"{all_results[best_name]['accuracy']:.1%}"
)

st.sidebar.metric(
    "Dataset Size",
    f"{len(df)} Patients"
)

st.sidebar.info(
    "This tool is for educational purposes and should not replace professional medical advice."
)
