import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import bcrypt

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

conn = sqlite3.connect(
    "users.db",
    check_same_thread=False
)

cursor = conn.cursor()


# ============================================================
# CREATE USERS TABLE
# ============================================================

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password BLOB NOT NULL
    )
    """
)


# ============================================================
# CREATE PREDICTIONS TABLE
# ============================================================

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        prediction_date TEXT NOT NULL,
        prediction TEXT NOT NULL,
        probability REAL NOT NULL,
        age INTEGER,
        sex TEXT,
        height REAL,
        weight REAL,
        bmi REAL,
        cholesterol INTEGER,
        blood_pressure INTEGER,
        model TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """
)

conn.commit()


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "username" not in st.session_state:
    st.session_state.username = None


# ============================================================
# REGISTER USER
# ============================================================

def register_user(username, email, password):

    if not username or not email or not password:
        return False, "Please complete all fields."

    hashed_password = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    )

    try:

        cursor.execute(
            """
            INSERT INTO users
            (
                username,
                email,
                password
            )
            VALUES (?, ?, ?)
            """,
            (
                username,
                email,
                hashed_password
            )
        )

        conn.commit()

        return True, "Account created successfully."

    except sqlite3.IntegrityError:

        return False, "Username or email already exists."

    except Exception as e:

        return False, f"Registration error: {e}"


# ============================================================
# LOGIN USER
# ============================================================

def login_user(username, password):

    cursor.execute(
        """
        SELECT
            id,
            username,
            email,
            password
        FROM users
        WHERE username = ?
        """,
        (username,)
    )

    user = cursor.fetchone()

    if user is None:
        return None

    stored_hash = user[3]

    try:

        if bcrypt.checkpw(
            password.encode("utf-8"),
            stored_hash
        ):

            return user

    except Exception:

        return None

    return None


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("heart.csv")

    # --------------------------------------------------------
    # CLEAN COLUMN NAMES
    # --------------------------------------------------------

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # CLEAN TEXT VALUES
    # --------------------------------------------------------

    for column in df.columns:

        if df[column].dtype == "object":

            df[column] = (
                df[column]
                .astype(str)
                .str.strip()
                .str.lower()
            )

    # --------------------------------------------------------
    # CONVERT THAL
    # --------------------------------------------------------

    if "thal" in df.columns:

        thal_mapping = {
            "normal": 0,
            "fixed": 1,
            "reversible": 2,
            "fixed defect": 1,
            "reversible defect": 2,
            "unknown": 3,
            "nan": np.nan
        }

        df["thal"] = df["thal"].map(
            thal_mapping
        )

    # --------------------------------------------------------
    # CONVERT SEX
    # --------------------------------------------------------

    if "sex" in df.columns:

        df["sex"] = df["sex"].replace(
            {
                "female": 0,
                "male": 1
            }
        )

    # --------------------------------------------------------
    # CONVERT FBS
    # --------------------------------------------------------

    if "fbs" in df.columns:

        df["fbs"] = df["fbs"].replace(
            {
                "no": 0,
                "yes": 1
            }
        )

    # --------------------------------------------------------
    # CONVERT EXANG
    # --------------------------------------------------------

    if "exang" in df.columns:

        df["exang"] = df["exang"].replace(
            {
                "no": 0,
                "yes": 1
            }
        )

    # --------------------------------------------------------
    # CONVERT NUMERIC COLUMNS
    # --------------------------------------------------------

    for column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # REMOVE INVALID ROWS
    # --------------------------------------------------------

    df = df.dropna()

    df = df.astype(float)

    return df


# ============================================================
# TRAIN MACHINE LEARNING MODELS
# ============================================================

@st.cache_resource
def train_models(_df):

    X = _df.drop(
        "target",
        axis=1
    )

    y = _df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

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

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        results[name] = {
            "model": model,
            "accuracy": accuracy
        }

    return results, X.columns.tolist()


# ============================================================
# LOAD DATASET
# ============================================================

try:

    df = load_data()

except FileNotFoundError:

    st.error(
        "The file 'heart.csv' could not be found. "
        "Please place heart.csv in the same folder as app.py."
    )

    st.stop()

except Exception as e:

    st.error(
        f"Error loading heart.csv: {e}"
    )

    st.stop()


# ============================================================
# REQUIRED COLUMNS
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
        "The following columns are missing from heart.csv: "
        + ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# TRAIN MODELS
# ============================================================

try:

    all_results, feature_names = train_models(
        df
    )

except Exception as e:

    st.error(
        f"Error training the machine learning models: {e}"
    )

    st.stop()


# ============================================================
# BEST MODEL
# ============================================================

best_name = max(
    all_results,
    key=lambda name: all_results[name]["accuracy"]
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("❤️ Heart Disease Predictor")

if st.session_state.logged_in:

    st.sidebar.success(
        f"Welcome, {st.session_state.username}"
    )

    menu = st.sidebar.radio(
        "Dashboard",
        [
            "New Assessment",
            "My Records",
            "About",
            "Logout"
        ]
    )

else:

    menu = st.sidebar.radio(
        "Menu",
        [
            "Sign In",
            "Sign Up",
            "About"
        ]
    )


# ============================================================
# ABOUT SECTION
# ============================================================

if menu == "About":

    st.title("About the Application")

    st.write(
        """
        The Heart Disease Risk Predictor is an educational
        machine-learning application designed to demonstrate
        how patient information can be used to estimate
        cardiovascular risk.
        """
    )

    st.info(
        """
        Important: This application is for educational
        purposes only. It is not a medical diagnostic tool
        and should not replace professional medical advice.
        """
    )

    # ========================================================
    # DATASET ANALYTICS
    # ========================================================

    st.header("Dataset Analytics")

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:

        st.metric(
            "Patient Records",
            len(df)
        )

    with metric2:

        st.metric(
            "Features",
            len(feature_names)
        )

    with metric3:

        positive_cases = int(
            df["target"].sum()
        )

        st.metric(
            "Positive Cases",
            positive_cases
        )

    with metric4:

        negative_cases = int(
            len(df) - df["target"].sum()
        )

        st.metric(
            "Negative Cases",
            negative_cases
        )

    # ========================================================
    # TARGET DISTRIBUTION
    # ========================================================

    st.subheader("Target Distribution")

    target_counts = df["target"].value_counts().sort_index()

    target_df = pd.DataFrame(
        {
            "Outcome": [
                "No Heart Disease",
                "Heart Disease"
            ],
            "Patients": [
                target_counts.get(0, 0),
                target_counts.get(1, 0)
            ]
        }
    )

    st.bar_chart(
        target_df.set_index("Outcome")
    )

    # ========================================================
    # DATASET PREVIEW
    # ========================================================

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.header("Model Performance")

    st.write(
        "The application trains and compares three machine-learning models using the same dataset."
    )

    performance_data = []

    for name, result in all_results.items():

        performance_data.append(
            {
                "Model": name,
                "Accuracy": result["accuracy"]
            }
        )

    performance_df = pd.DataFrame(
        performance_data
    )

    performance_df["Accuracy"] = (
        performance_df["Accuracy"] * 100
    ).round(1)

    st.dataframe(
        performance_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # MODEL DETAILS
    # ========================================================

    st.subheader("Model Details")

    st.write(
        "**Random Forest**"
    )

    st.write(
        "An ensemble classification model that combines multiple decision trees to make predictions."
    )

    st.write(
        "**Gradient Boosting**"
    )

    st.write(
        "A sequential ensemble model that builds multiple weak learners to improve prediction performance."
    )

    st.write(
        "**Logistic Regression**"
    )

    st.write(
        "A statistical classification model commonly used for binary prediction problems."
    )

    st.success(
        f"Best performing model: {best_name} "
        f"({all_results[best_name]['accuracy']:.1%} accuracy)"
    )

    # ========================================================
    # FEATURES
    # ========================================================

    st.header("Prediction Features")

    feature_descriptions = {
        "age": "Patient age",
        "sex": "Patient sex",
        "cp": "Chest pain type",
        "trestbps": "Resting blood pressure",
        "chol": "Serum cholesterol",
        "fbs": "Fasting blood sugar",
        "restecg": "Resting ECG results",
        "thalach": "Maximum heart rate achieved",
        "exang": "Exercise-induced angina",
        "oldpeak": "ST depression",
        "slope": "Slope of peak exercise ST segment",
        "ca": "Number of major vessels",
        "thal": "Thalassemia"
    }

    feature_table = pd.DataFrame(
        {
            "Feature": feature_names,
            "Description": [
                feature_descriptions.get(
                    feature,
                    "Clinical feature"
                )
                for feature in feature_names
            ]
        }
    )

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SIGN IN PAGE
# ============================================================

elif menu == "Sign In" and not st.session_state.logged_in:

    st.title("❤️ Heart Disease Risk Predictor")

    st.subheader("Welcome Back")

    st.write(
        "Sign in to access your health assessments and prediction history."
    )

    st.divider()

    username = st.text_input(
        "Username",
        key="login_username"
    )

    password = st.text_input(
        "Password",
        type="password",
        key="login_password"
    )

    if st.button(
        "Sign In",
        use_container_width=True,
        type="primary"
    ):

        user = login_user(
            username,
            password
        )

        if user:

            st.session_state.logged_in = True
            st.session_state.user_id = user[0]
            st.session_state.username = user[1]

            st.rerun()

        else:

            st.error(
                "Invalid username or password."
            )

    st.divider()

    st.caption(
        "Don't have an account? Select 'Sign Up' from the sidebar."
    )


# ============================================================
# SIGN UP PAGE
# ============================================================

elif menu == "Sign Up" and not st.session_state.logged_in:

    st.title("Create Your Account")

    st.write(
        "Create an account to save and track your assessments."
    )

    st.divider()

    username = st.text_input(
        "Username",
        key="register_username"
    )

    email = st.text_input(
        "Email",
        key="register_email"
    )

    password = st.text_input(
        "Password",
        type="password",
        key="register_password"
    )

    confirm_password = st.text_input(
        "Confirm Password",
        type="password",
        key="register_confirm"
    )

    if st.button(
        "Create Account",
        use_container_width=True,
        type="primary"
    ):

        if password != confirm_password:

            st.error(
                "Passwords do not match."
            )

        else:

            success, message = register_user(
                username,
                email,
                password
            )

            if success:

                st.success(
                    "Account created successfully."
                )

                st.info(
                    "You can now select 'Sign In' from the sidebar."
                )

            else:

                st.error(
                    message
                )


# ============================================================
# NEW ASSESSMENT
# ============================================================

elif menu == "New Assessment" and st.session_state.logged_in:

    st.title("Heart Disease Risk Predictor")

    st.write(
        "Enter patient information below to estimate heart disease risk."
    )

    st.info(
        "This tool is for educational purposes only and should not replace professional medical advice."
    )

    # ========================================================
    # MODEL SELECTION
    # ========================================================

    st.sidebar.divider()

    st.sidebar.header(
        "Model Selection"
    )

    comparison_df = pd.DataFrame(
        {
            "Model": list(all_results.keys()),
            "Accuracy": [
                f"{all_results[name]['accuracy']:.1%}"
                for name in all_results
            ]
        }
    )

    st.sidebar.dataframe(
        comparison_df,
        hide_index=True
    )

    st.sidebar.write(
        f"Best performer: {best_name}"
    )

    selected_model_name = st.sidebar.selectbox(
        "Choose Prediction Model",
        options=list(all_results.keys()),
        index=list(
            all_results.keys()
        ).index(best_name)
    )

    model = all_results[
        selected_model_name
    ]["model"]

    accuracy = all_results[
        selected_model_name
    ]["accuracy"]

    st.sidebar.metric(
        "Selected Model Accuracy",
        f"{accuracy:.1%}"
    )

    st.sidebar.write(
        f"Trained on {len(df)} patient records"
    )

    # ========================================================
    # PATIENT INFORMATION
    # ========================================================

    st.header("Patient Information")

    col1, col2 = st.columns(2)

    # ========================================================
    # COLUMN 1
    # ========================================================

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
            "Female"
            if x == 0
            else "Male"
        )

        height = st.number_input(
            "Height (cm)",
            min_value=50.0,
            max_value=250.0,
            value=170.0,
            step=0.5
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=300.0,
            value=70.0,
            step=0.5
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
            "No"
            if x == 0
            else "Yes"
        )

    # ========================================================
    # COLUMN 2
    # ========================================================

    with col2:

        restecg = st.selectbox(
            "Resting ECG Results",
            options=[0, 1, 2],
            format_func=lambda x: [
                "Normal",
                "ST-T Abnormality",
                "Left Ventricular Hypertrophy"
            ][x]
        )

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
            "No"
            if x == 0
            else "Yes"
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

    # ========================================================
    # BMI
    # ========================================================

    bmi = weight / ((height / 100) ** 2)

    st.metric(
        "Calculated BMI",
        f"{bmi:.1f}"
    )

    # ========================================================
    # PREDICT
    # ========================================================

    if st.button(
        "Predict Risk",
        type="primary",
        use_container_width=True
    ):

        # ----------------------------------------------------
        # CREATE INPUT DATA
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

        # ----------------------------------------------------
        # PREDICTION LABEL
        # ----------------------------------------------------

        if prediction == 1:

            prediction_label = "Positive"

        else:

            prediction_label = "Negative"

        # ----------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------

        if probability < 0.30:

            risk_level = "Low Risk"

        elif probability < 0.70:

            risk_level = "Moderate Risk"

        else:

            risk_level = "High Risk"

        # ----------------------------------------------------
        # SAVE TO DATABASE
        # ----------------------------------------------------

        cursor.execute(
            """
            INSERT INTO predictions
            (
                user_id,
                prediction_date,
                prediction,
                probability,
                age,
                sex,
                height,
                weight,
                bmi,
                cholesterol,
                blood_pressure,
                model
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                st.session_state.user_id,
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                risk_level,
                float(probability),
                int(age),
                "Female" if sex == 0 else "Male",
                float(height),
                float(weight),
                float(bmi),
                int(chol),
                int(trestbps),
                selected_model_name
            )
        )

        conn.commit()

        # ====================================================
        # RESULTS
        # ====================================================

        st.divider()

        st.header("Prediction Results")

        st.caption(
            f"Using: {selected_model_name}"
        )

        if risk_level == "Low Risk":

            st.success(
                f"{risk_level} of Heart Disease "
                f"({probability:.1%} probability)"
            )

        elif risk_level == "Moderate Risk":

            st.warning(
                f"{risk_level} of Heart Disease "
                f"({probability:.1%} probability)"
            )

        else:

            st.error(
                f"{risk_level} of Heart Disease "
                f"({probability:.1%} probability)"
            )

        # ====================================================
        # RISK LEVEL
        # ====================================================

        st.subheader("Risk Level")

        st.progress(
            int(probability * 100)
        )

        # ====================================================
        # METRICS
        # ====================================================

        metric1, metric2, metric3 = st.columns(3)

        with metric1:

            st.metric(
                "Risk Probability",
                f"{probability:.1%}"
            )

        with metric2:

            st.metric(
                "Prediction",
                prediction_label
            )

        with metric3:

            st.metric(
                "Model Accuracy",
                f"{accuracy:.1%}"
            )

        # ====================================================
        # HEALTH INFORMATION
        # ====================================================

        st.subheader("Patient Summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:

            st.metric(
                "Age",
                f"{age} years"
            )

        with summary_col2:

            st.metric(
                "BMI",
                f"{bmi:.1f}"
            )

        with summary_col3:

            st.metric(
                "Cholesterol",
                f"{chol} mg/dl"
            )

        # ====================================================
        # HEALTH RECOMMENDATIONS
        # ====================================================

        st.subheader(
            "General Health Recommendations"
        )

        if probability < 0.30:

            st.info(
                """
                Maintain regular physical activity.

                Maintain a balanced diet.

                Continue monitoring blood pressure and cholesterol.

                Attend routine health check-ups.
                """
            )

        elif probability < 0.70:

            st.warning(
                """
                Consider increasing physical activity.

                Maintain a balanced diet and reduce excess saturated fats and sodium.

                Monitor blood pressure and cholesterol.

                Consider discussing the result with a healthcare professional.
                """
            )

        else:

            st.error(
                """
                Consider seeking professional medical advice.

                Monitor blood pressure and cholesterol closely.

                Follow any treatment recommendations provided by a healthcare professional.

                Do not use this prediction as a medical diagnosis.
                """
            )

        # ====================================================
        # HEALTH ALERTS
        # ====================================================

        st.subheader(
            "Health Alerts"
        )

        alerts_found = False

        if trestbps > 140:

            st.warning(
                "The entered resting blood pressure is above 140 mm Hg."
            )

            alerts_found = True

        if chol > 240:

            st.warning(
                "The entered cholesterol level is above 240 mg/dl."
            )

            alerts_found = True

        if exang == 1:

            st.warning(
                "Exercise-induced angina was reported."
            )

            alerts_found = True

        if oldpeak > 2:

            st.warning(
                "The entered ST depression is above 2."
            )

            alerts_found = True

        if bmi >= 30:

            st.warning(
                f"The calculated BMI is {bmi:.1f}."
            )

            alerts_found = True

        if not alerts_found:

            st.success(
                "No additional alerts were triggered by the entered values."
            )

        # ====================================================
        # TOP RISK FACTORS
        # ====================================================

        st.subheader(
            "Top Risk Factors"
        )

        if hasattr(
            model,
            "feature_importances_"
        ):

            importance_values = model.feature_importances_

        else:

            importance_values = np.abs(
                model.coef_[0]
            )

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importance_values
            }
        )

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        ).head(10)

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.barh(
            importance_df["Feature"],
            importance_df["Importance"]
        )

        ax.set_title(
            "Top Model Risk Factors"
        )

        ax.set_xlabel(
            "Importance"
        )

        ax.invert_yaxis()

        st.pyplot(
            fig
        )

        plt.close(fig)

        # ====================================================
        # DOWNLOAD REPORT
        # ====================================================

        st.subheader(
            "Download Report"
        )

        report = f"""
HEART DISEASE RISK REPORT
=========================

Patient
-------

Username: {st.session_state.username}

Date:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


PATIENT INFORMATION
-------------------

Age: {age}

Sex: {"Female" if sex == 0 else "Male"}

Height: {height:.1f} cm

Weight: {weight:.1f} kg

BMI: {bmi:.1f}

Cholesterol: {chol} mg/dl

Blood Pressure: {trestbps} mm Hg


PREDICTION RESULTS
------------------

Prediction: {prediction_label}

Risk Level: {risk_level}

Risk Probability: {probability:.2%}

Model Used: {selected_model_name}

Model Accuracy: {accuracy:.2%}


IMPORTANT
---------

This report is generated by an educational
machine-learning application.

It should not replace professional medical advice.
"""

        st.download_button(
            label="Download Risk Report",
            data=report,
            file_name="heart_risk_report.txt",
            mime="text/plain"
        )


# ============================================================
# MY RECORDS
# ============================================================

elif menu == "My Records" and st.session_state.logged_in:

    st.title("My Records")

    st.write(
        "View your previous heart disease assessments."
    )

    records_df = pd.read_sql_query(
        """
        SELECT
            prediction_date AS Date,
            prediction AS Risk_Level,
            probability AS Probability,
            age AS Age,
            sex AS Sex,
            height AS Height_cm,
            weight AS Weight_kg,
            bmi AS BMI,
            cholesterol AS Cholesterol,
            blood_pressure AS Blood_Pressure,
            model AS Model

        FROM predictions

        WHERE user_id = ?

        ORDER BY prediction_date DESC
        """,
        conn,
        params=(
            st.session_state.user_id,
        )
    )

    if records_df.empty:

        st.info(
            "You do not have any prediction records yet."
        )

    else:

        records_df["Probability"] = (
            records_df["Probability"] * 100
        ).round(1)

        records_df["BMI"] = (
            records_df["BMI"]
            .round(1)
        )

        records_df["Height_cm"] = (
            records_df["Height_cm"]
            .round(1)
        )

        records_df["Weight_kg"] = (
            records_df["Weight_kg"]
            .round(1)
        )

        st.dataframe(
            records_df,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        st.subheader(
            "Assessment Summary"
        )

        total_records = len(
            records_df
        )

        high_risk = len(
            records_df[
                records_df["Risk_Level"] == "High Risk"
            ]
        )

        moderate_risk = len(
            records_df[
                records_df["Risk_Level"] == "Moderate Risk"
            ]
        )

        low_risk = len(
            records_df[
                records_df["Risk_Level"] == "Low Risk"
            ]
        )

        metric1, metric2, metric3, metric4 = st.columns(4)

        with metric1:

            st.metric(
                "Assessments",
                total_records
            )

        with metric2:

            st.metric(
                "Low Risk",
                low_risk
            )

        with metric3:

            st.metric(
                "Moderate Risk",
                moderate_risk
            )

        with metric4:

            st.metric(
                "High Risk",
                high_risk
            )

        # ====================================================
        # DOWNLOAD RECORDS
        # ====================================================

        csv_data = records_df.to_csv(
            index=False
        )

        st.download_button(
            label="Download My Records",
            data=csv_data,
            file_name="my_heart_disease_records.csv",
            mime="text/csv"
        )


# ============================================================
# LOGOUT
# ============================================================

elif menu == "Logout" and st.session_state.logged_in:

    st.session_state.logged_in = False

    st.session_state.user_id = None

    st.session_state.username = None

    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "❤️ Heart Disease Risk Predictor | Educational Machine Learning Application"
)

st.caption(
    "This application does not provide medical diagnoses. "
    "Consult a qualified healthcare professional for medical advice."
)
