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
    box-shadow: 0 10px 30px rgba(0,0,0,.15);
}

.logo-circle {
    width: 75px;
    height: 75px;
    background: white;
    border-radius: 50%;
    margin: 0 auto 20px auto;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 32px;
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

.section-card {
    background: #F8FAFC;
    padding: 25px;
    border-radius: 15px;
    margin: 15px 0;
    border: 1px solid #E2E8F0;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #E2E8F0;
}

.metric-number {
    font-size: 28px;
    font-weight: 700;
    color: #1E3A8A;
}

.metric-label {
    font-size: 14px;
    color: #64748B;
}

.model-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #E2E8F0;
    margin-bottom: 12px;
}

.model-name {
    font-size: 18px;
    font-weight: 600;
    color: #0F172A;
}

.model-score {
    font-size: 26px;
    font-weight: 700;
    color: #2563EB;
}

.step-card {
    background: #F8FAFC;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #E2E8F0;
    min-height: 150px;
}

.step-number {
    font-size: 28px;
    font-weight: 700;
    color: #2563EB;
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
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("heart.csv")

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Clean text values
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
    # Convert THAL
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
    # Convert SEX
    # --------------------------------------------------------

    if "sex" in df.columns:

        df["sex"] = df["sex"].replace(
            {
                "female": 0,
                "male": 1
            }
        )

    # --------------------------------------------------------
    # Convert FBS
    # --------------------------------------------------------

    if "fbs" in df.columns:

        df["fbs"] = df["fbs"].replace(
            {
                "no": 0,
                "yes": 1
            }
        )

    # --------------------------------------------------------
    # Convert EXANG
    # --------------------------------------------------------

    if "exang" in df.columns:

        df["exang"] = df["exang"].replace(
            {
                "no": 0,
                "yes": 1
            }
        )

    # --------------------------------------------------------
    # Convert remaining columns
    # --------------------------------------------------------

    for column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove invalid rows
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
    key=lambda name: all_results[name]["accuracy"]
)


# ============================================================
# DATASET ANALYTICS
# ============================================================

total_patients = len(df)

total_features = len(
    df.drop(
        "target",
        axis=1
    ).columns
)

average_age = df["age"].mean()

average_cholesterol = df["chol"].mean()

average_blood_pressure = df["trestbps"].mean()

positive_cases = int(
    (df["target"] == 1).sum()
)

negative_cases = int(
    (df["target"] == 0).sum()
)

positive_percentage = (
    positive_cases / total_patients
) * 100

negative_percentage = (
    negative_cases / total_patients
) * 100


# ============================================================
# PUBLIC LANDING PAGE
# ============================================================

if not st.session_state.logged_in:

    # ========================================================
    # HERO
    # ========================================================

    st.markdown(
        """
<div class="hero-card">

    <div class="logo-circle">
        ❤️
    </div>

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
    # INTRODUCTION
    # ========================================================

    st.subheader(
        "About the Application"
    )

    st.write(
        """
        Heart Disease Risk Predictor is a machine-learning application
        designed to estimate cardiovascular risk based on patient health
        information.

        The application analyses patient data using multiple machine-learning
        models and provides a risk probability, risk classification,
        health alerts and personalised recommendations.

        This application is intended for educational and demonstration
        purposes and should not be used as a medical diagnostic tool.
        """
    )


    # ========================================================
    # DATASET ANALYTICS
    # ========================================================

    st.divider()

    st.header(
        "Dataset Analytics"
    )

    st.write(
        "Overview of the dataset used to train and evaluate the machine-learning models."
    )


    # --------------------------------------------------------
    # DATASET METRICS
    # --------------------------------------------------------

    metric1, metric2, metric3, metric4 = st.columns(4)


    with metric1:

        st.metric(
            "Patient Records",
            f"{total_patients:,}"
        )


    with metric2:

        st.metric(
            "Features",
            total_features
        )


    with metric3:

        st.metric(
            "Average Age",
            f"{average_age:.1f}"
        )


    with metric4:

        st.metric(
            "Average Cholesterol",
            f"{average_cholesterol:.1f}"
        )


    metric5, metric6 = st.columns(2)


    with metric5:

        st.metric(
            "Average Blood Pressure",
            f"{average_blood_pressure:.1f} mm Hg"
        )


    with metric6:

        st.metric(
            "Positive Cases",
            f"{positive_cases:,}"
        )


    # ========================================================
    # TARGET DISTRIBUTION
    # ========================================================

    st.subheader(
        "Heart Disease Target Distribution"
    )

    target_col1, target_col2 = st.columns(2)


    with target_col1:

        st.write(
            f"**Positive cases:** {positive_cases:,} "
            f"({positive_percentage:.1f}%)"
        )

        st.write(
            f"**Negative cases:** {negative_cases:,} "
            f"({negative_percentage:.1f}%)"
        )


    with target_col2:

        target_counts = pd.DataFrame(
            {
                "Outcome": [
                    "Negative",
                    "Positive"
                ],
                "Patients": [
                    negative_cases,
                    positive_cases
                ]
            }
        )

        fig_target, ax_target = plt.subplots(
            figsize=(5, 3)
        )

        ax_target.bar(
            target_counts["Outcome"],
            target_counts["Patients"]
        )

        ax_target.set_ylabel(
            "Number of Patients"
        )

        ax_target.set_title(
            "Target Distribution"
        )

        st.pyplot(
            fig_target
        )

        plt.close(
            fig_target
        )


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.divider()

    st.header(
        "Model Performance"
    )

    st.write(
        "The application trains and compares three machine-learning models using the same dataset."
    )


    model_col1, model_col2, model_col3 = st.columns(3)


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    with model_col1:

        rf_accuracy = all_results[
            "Random Forest"
        ]["accuracy"]

        st.markdown(
            f"""
<div class="model-card">

    <div class="model-name">
        Random Forest
    </div>

    <div class="model-score">
        {rf_accuracy:.1%}
    </div>

    <p>
        Ensemble classification model using multiple decision trees.
    </p>

</div>
""",
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # GRADIENT BOOSTING
    # --------------------------------------------------------

    with model_col2:

        gb_accuracy = all_results[
            "Gradient Boosting"
        ]["accuracy"]

        st.markdown(
            f"""
<div class="model-card">

    <div class="model-name">
        Gradient Boosting
    </div>

    <div class="model-score">
        {gb_accuracy:.1%}
    </div>

    <p>
        Sequential ensemble model designed to improve prediction performance.
    </p>

</div>
""",
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    with model_col3:

        lr_accuracy = all_results[
            "Logistic Regression"
        ]["accuracy"]

        st.markdown(
            f"""
<div class="model-card">

    <div class="model-name">
        Logistic Regression
    </div>

    <div class="model-score">
        {lr_accuracy:.1%}
    </div>

    <p>
        Statistical classification model used for binary prediction.
    </p>

</div>
""",
            unsafe_allow_html=True
        )


    st.success(
        f"Best-performing model: {best_name} "
        f"with an accuracy of {all_results[best_name]['accuracy']:.1%}."
    )


    # ========================================================
    # MODEL COMPARISON CHART
    # ========================================================

    st.subheader(
        "Model Accuracy Comparison"
    )

    model_names = list(
        all_results.keys()
    )

    model_accuracies = [
        all_results[name]["accuracy"] * 100
        for name in model_names
    ]

    comparison_df = pd.DataFrame(
        {
            "Model": model_names,
            "Accuracy": model_accuracies
        }
    )

    fig_models, ax_models = plt.subplots(
        figsize=(8, 4)
    )

    ax_models.bar(
        comparison_df["Model"],
        comparison_df["Accuracy"]
    )

    ax_models.set_ylabel(
        "Accuracy (%)"
    )

    ax_models.set_ylim(
        0,
        100
    )

    ax_models.set_title(
        "Machine Learning Model Performance"
    )

    for index, value in enumerate(model_accuracies):

        ax_models.text(
            index,
            value + 1,
            f"{value:.1f}%",
            ha="center"
        )

    st.pyplot(
        fig_models
    )

    plt.close(
        fig_models
    )


    # ========================================================
    # HOW IT WORKS
    # ========================================================

    st.divider()

    st.header(
        "How It Works"
    )

    step1, step2, step3, step4 = st.columns(4)


    with step1:

        st.markdown(
            """
<div class="step-card">

    <div class="step-number">
        1
    </div>

    <h4>Enter Information</h4>

    <p>
        Provide patient health information.
    </p>

</div>
""",
            unsafe_allow_html=True
        )


    with step2:

        st.markdown(
            """
<div class="step-card">

    <div class="step-number">
        2
    </div>

    <h4>AI Analysis</h4>

    <p>
        The selected machine-learning model analyses the information.
    </p>

</div>
""",
            unsafe_allow_html=True
        )


    with step3:

        st.markdown(
            """
<div class="step-card">

    <div class="step-number">
        3
    </div>

    <h4>Risk Prediction</h4>

    <p>
        The system calculates an estimated risk probability.
    </p>

</div>
""",
            unsafe_allow_html=True
        )


    with step4:

        st.markdown(
            """
<div class="step-card">

    <div class="step-number">
        4
    </div>

    <h4>View Results</h4>

    <p>
        Review the risk level, alerts and recommendations.
    </p>

</div>
""",
            unsafe_allow_html=True
        )


    # ========================================================
    # SIGN IN / SIGN UP
    # ========================================================

    st.divider()

    st.header(
        "Get Started"
    )

    st.write(
        "Create an account or sign in to perform a heart disease risk assessment and save your results."
    )


    tab1, tab2 = st.tabs(
        [
            "Sign In",
            "Create Account"
        ]
    )


    # ========================================================
    # SIGN IN
    # ========================================================

    with tab1:

        st.subheader(
            "Welcome Back"
        )

        st.write(
            "Sign in to access your assessments and prediction history."
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

        st.subheader(
            "Create Your Account"
        )

        st.write(
            "Create an account to save and track your assessments."
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
    # SIDEBAR
    # ========================================================

    st.sidebar.title(
        "Heart Disease Predictor"
    )


    st.sidebar.success(
        f"Welcome, {st.session_state.username}"
    )


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

        st.title(
            "Heart Disease Risk Predictor"
        )


        st.write(
            "Enter patient information below to estimate heart disease risk."
        )


        st.info(
            "This tool is for educational purposes only and should not replace professional medical advice."
        )


        # ----------------------------------------------------
        # MODEL SELECTOR
        # ----------------------------------------------------

        st.sidebar.divider()

        st.sidebar.header(
            "Prediction Model"
        )


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
            "Model Accuracy",
            f"{accuracy:.1%}"
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
                    cholesterol,
                    blood_pressure,
                    model
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            # RISK LEVEL
            # ------------------------------------------------

            st.subheader(
                "Risk Probability"
            )


            st.progress(
                int(probability * 100)
            )


            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

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


            st.dataframe(
                records_df,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

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


            # ------------------------------------------------
            # DOWNLOAD RECORDS
            # ------------------------------------------------

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
# SIDEBAR FOOTER
# ============================================================

if st.session_state.logged_in:

    st.sidebar.divider()

    st.sidebar.info(
        "This application is for educational purposes and should not replace professional medical advice."
    )
