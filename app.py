import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import bcrypt
import base64

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
    page_icon="logo.png",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.hero-card {
    background: linear-gradient(135deg, #0F172A, #1E3A8A);
    padding: 45px 30px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
}

.hero-logo {
    width: 90px;
    height: 90px;
    object-fit: contain;
    background: white;
    border-radius: 50%;
    padding: 10px;
    margin-bottom: 20px;
}

.hero-title {
    color: white;
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 10px;
}

.hero-subtitle {
    color: #DBEAFE;
    font-size: 17px;
}

</style>
""",
    unsafe_allow_html=True
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
# ADD NEW COLUMNS IF DATABASE ALREADY EXISTS
# ============================================================

existing_columns = [
    row[1]
    for row in cursor.execute(
        "PRAGMA table_info(predictions)"
    ).fetchall()
]

if "height" not in existing_columns:
    cursor.execute(
        "ALTER TABLE predictions ADD COLUMN height REAL"
    )

if "weight" not in existing_columns:
    cursor.execute(
        "ALTER TABLE predictions ADD COLUMN weight REAL"
    )

if "bmi" not in existing_columns:
    cursor.execute(
        "ALTER TABLE predictions ADD COLUMN bmi REAL"
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

    if len(password) < 6:
        return False, "Password must contain at least 6 characters."

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
    # CONVERT REMAINING COLUMNS
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
# TRAIN MODELS
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
# LOAD HEART DISEASE DATA
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
    key=lambda name:
    all_results[name]["accuracy"]
)


# ============================================================
# LANDING PAGE
# ============================================================

if not st.session_state.logged_in:

    # ========================================================
    # SIDEBAR ABOUT SECTION
    # ========================================================

    st.sidebar.title(
        "Heart Disease Predictor"
    )

    with st.sidebar.expander(
        "About",
        expanded=True
    ):

        st.write(
            """
            **Heart Disease Risk Predictor**

            This application is an educational machine-learning
            system designed to demonstrate how patient health
            information can be used to estimate cardiovascular risk.
            """
        )

        st.write("### How It Works")

        st.write(
            """
            1. Create an account or sign in.
            2. Enter patient information.
            3. Select a machine-learning model.
            4. Generate a risk prediction.
            5. Review the results and risk factors.
            6. Save and review previous assessments.
            """
        )

        st.write("### Machine Learning Models")

        for model_name in all_results:

            accuracy = all_results[
                model_name
            ]["accuracy"]

            st.write(
                f"**{model_name}** — {accuracy:.1%}"
            )

        st.write("### Dataset")

        st.write(
            f"""
            The application uses a heart disease dataset containing
            **{len(df)} patient records**.

            The dataset includes cardiovascular indicators such as
            age, blood pressure, cholesterol, heart rate, chest pain,
            exercise-induced angina and other clinical attributes.
            """
        )

        st.write("### Important")

        st.warning(
            "This application is for educational purposes only "
            "and should not replace professional medical advice."
        )


    # ========================================================
    # HERO SECTION
    # ========================================================

    try:

        with open("logo.png", "rb") as image_file:

            encoded_logo = base64.b64encode(
                image_file.read()
            ).decode()

        logo_html = f"""
        <img
            src="data:image/png;base64,{encoded_logo}"
            class="hero-logo"
        />
        """

    except Exception:

        logo_html = ""


    st.markdown(
        f"""
        <div class="hero-card">

            {logo_html}

            <div class="hero-title">
                Heart Disease Risk Predictor
            </div>

            <div class="hero-subtitle">
                AI-Powered Cardiovascular Risk Assessment
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # SIGN IN / SIGN UP TABS
    # ========================================================

    tab1, tab2 = st.tabs(
        [
            "Sign In",
            "Sign Up"
        ]
    )


    # ========================================================
    # SIGN IN
    # ========================================================

    with tab1:

        st.header(
            "Welcome Back"
        )

        st.write(
            "Sign in to access your health assessments "
            "and prediction history."
        )

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

            if not username or not password:

                st.error(
                    "Please enter your username and password."
                )

            else:

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


    # ========================================================
    # SIGN UP
    # ========================================================

    with tab2:

        st.header(
            "Create Your Account"
        )

        st.write(
            "Create an account to save and track your "
            "heart disease assessments."
        )

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
                        "You can now sign in using the Sign In tab."
                    )

                else:

                    st.error(
                        message
                    )


# ============================================================
# LOGGED-IN APPLICATION
# ============================================================

else:

    # ========================================================
    # SIDEBAR USER INFORMATION
    # ========================================================

    st.sidebar.success(
        f"Welcome, {st.session_state.username}"
    )


    # ========================================================
    # SIDEBAR NAVIGATION
    # ========================================================

    menu = st.sidebar.radio(
        "Dashboard",
        [
            "New Assessment",
            "My Records",
            "Logout"
        ]
    )


    # ========================================================
    # NEW ASSESSMENT
    # ========================================================

    if menu == "New Assessment":

        # ----------------------------------------------------
        # MODEL COMPARISON
        # ----------------------------------------------------

        st.sidebar.divider()

        st.sidebar.header(
            "Model Comparison"
        )

        comparison_df = pd.DataFrame(
            {
                "Model": list(
                    all_results.keys()
                ),

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


        # ----------------------------------------------------
        # MODEL SELECTOR
        # ----------------------------------------------------

        selected_model_name = st.sidebar.selectbox(
            "Choose Prediction Model",
            options=list(
                all_results.keys()
            ),
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


        # ====================================================
        # MAIN CONTENT
        # ====================================================

        st.title(
            "Heart Disease Risk Predictor"
        )

        st.write(
            "Enter patient information below to predict "
            "heart disease risk."
        )

        st.info(
            "This tool is for educational purposes only "
            "and should not replace professional medical advice."
        )


        # ====================================================
        # PATIENT INFORMATION
        # ====================================================

        st.header(
            "Patient Information"
        )

        col1, col2 = st.columns(2)


        # ====================================================
        # COLUMN 1
        # ====================================================

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
                step=1.0
            )

            weight = st.number_input(
                "Weight (kg)",
                min_value=20.0,
                max_value=300.0,
                value=70.0,
                step=1.0
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

            restecg = st.selectbox(
                "Resting ECG Results",
                options=[0, 1, 2],
                format_func=lambda x: [
                    "Normal",
                    "ST-T Abnormality",
                    "Left Ventricular Hypertrophy"
                ][x]
            )


        # ====================================================
        # COLUMN 2
        # ====================================================

        with col2:

            # ------------------------------------------------
            # BMI CALCULATION
            # ------------------------------------------------

            height_m = height / 100

            bmi = weight / (
                height_m ** 2
            )

            st.metric(
                "Calculated BMI",
                f"{bmi:.1f}"
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


        # ====================================================
        # PREDICT
        # ====================================================

        if st.button(
            "Predict Risk",
            type="primary",
            use_container_width=True
        ):

            # ------------------------------------------------
            # CREATE INPUT DATA
            # ------------------------------------------------

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


            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            prediction = model.predict(
                input_data
            )[0]

            probability = model.predict_proba(
                input_data
            )[0][1]


            # ------------------------------------------------
            # PREDICTION LABEL
            # ------------------------------------------------

            if prediction == 1:

                prediction_label = "Positive"

            else:

                prediction_label = "Negative"


            # ------------------------------------------------
            # RISK LEVEL
            # ------------------------------------------------

            if probability < 0.30:

                risk_level = "Low Risk"

            elif probability < 0.70:

                risk_level = "Moderate Risk"

            else:

                risk_level = "High Risk"


            # ------------------------------------------------
            # SAVE TO DATABASE
            # ------------------------------------------------

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
                    "Female"
                    if sex == 0
                    else "Male",
                    float(height),
                    float(weight),
                    float(bmi),
                    int(chol),
                    int(trestbps),
                    selected_model_name
                )
            )

            conn.commit()


            # =================================================
            # RESULTS
            # =================================================

            st.divider()

            st.header(
                "Prediction Results"
            )

            st.caption(
                f"Using: {selected_model_name}"
            )


            # ------------------------------------------------
            # RISK MESSAGE
            # ------------------------------------------------

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


            # ------------------------------------------------
            # PROGRESS BAR
            # ------------------------------------------------

            st.subheader(
                "Risk Level"
            )

            st.progress(
                int(probability * 100)
            )


            # =================================================
            # METRICS
            # =================================================

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


            # =================================================
            # BMI INFORMATION
            # =================================================

            st.subheader(
                "Body Mass Index"
            )

            bmi_col1, bmi_col2 = st.columns(2)

            with bmi_col1:

                st.metric(
                    "BMI",
                    f"{bmi:.1f}"
                )

            with bmi_col2:

                if bmi < 18.5:

                    bmi_category = "Underweight"

                elif bmi < 25:

                    bmi_category = "Healthy Weight"

                elif bmi < 30:

                    bmi_category = "Overweight"

                else:

                    bmi_category = "Obesity"

                st.metric(
                    "BMI Category",
                    bmi_category
                )


            # =================================================
            # HEALTH RECOMMENDATIONS
            # =================================================

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


            # =================================================
            # HEALTH ALERTS
            # =================================================

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


            if bmi >= 30:

                st.warning(
                    "BMI is in the obesity range."
                )

                alerts_found = True


            elif bmi >= 25:

                st.info(
                    "BMI is in the overweight range."
                )


            if not alerts_found:

                st.success(
                    "No additional health alerts detected."
                )


            # =================================================
            # TOP RISK FACTORS
            # =================================================

            st.subheader(
                "Top Risk Factors"
            )

            if hasattr(
                model,
                "feature_importances_"
            ):

                importance_df = pd.DataFrame(
                    {
                        "Feature": feature_names,
                        "Importance": model.feature_importances_
                    }
                )

            else:

                importance_df = pd.DataFrame(
                    {
                        "Feature": feature_names,
                        "Importance": np.abs(
                            model.coef_[0]
                        )
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
                "Top Risk Factors"
            )


            ax.set_xlabel(
                "Importance"
            )


            ax.invert_yaxis()


            st.pyplot(
                fig
            )


            plt.close(
                fig
            )


            # =================================================
            # DOWNLOAD REPORT
            # =================================================

            st.subheader(
                "Download Report"
            )


            report = f"""
Heart Disease Risk Report
==========================

Patient
-------

Username: {st.session_state.username}

Date:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


Patient Information
-------------------

Age: {age}

Sex: {"Female" if sex == 0 else "Male"}

Height: {height:.1f} cm

Weight: {weight:.1f} kg

BMI: {bmi:.1f}

BMI Category: {bmi_category}

Cholesterol: {chol} mg/dl

Blood Pressure: {trestbps} mm Hg


Prediction Results
------------------

Prediction: {prediction_label}

Risk Level: {risk_level}

Risk Probability: {probability:.2%}

Model Used: {selected_model_name}

Model Accuracy: {accuracy:.2%}


Important
---------

This report is generated by an educational
machine learning application.

It should not replace professional medical advice.
"""


            st.download_button(
                label="Download Risk Report",
                data=report,
                file_name="heart_risk_report.txt",
                mime="text/plain"
            )


    # ========================================================
    # MY RECORDS
    # ========================================================

    elif menu == "My Records":

        st.header(
            "My Records"
        )

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

            records_df["Height_cm"] = (
                records_df["Height_cm"]
                .round(1)
            )

            records_df["Weight_kg"] = (
                records_df["Weight_kg"]
                .round(1)
            )

            records_df["BMI"] = (
                records_df["BMI"]
                .round(1)
            )


            st.dataframe(
                records_df,
                use_container_width=True,
                hide_index=True
            )


            # =================================================
            # SUMMARY
            # =================================================

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


            # =================================================
            # DOWNLOAD RECORDS
            # =================================================

            csv_data = records_df.to_csv(
                index=False
            )


            st.download_button(
                label="Download My Records",
                data=csv_data,
                file_name="my_heart_disease_records.csv",
                mime="text/csv"
            )


    # ========================================================
    # LOGOUT
    # ========================================================

    elif menu == "Logout":

        st.session_state.logged_in = False

        st.session_state.user_id = None

        st.session_state.username = None

        st.rerun()


# ============================================================
# LOGGED-IN SIDEBAR FOOTER
# ============================================================

if st.session_state.logged_in:

    st.sidebar.divider()

    st.sidebar.info(
        "This application is for educational purposes "
        "and should not replace professional medical advice."
    )
