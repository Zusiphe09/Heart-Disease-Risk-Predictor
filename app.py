import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import bcrypt
import io
import os

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# Optional packages
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image as ReportImage
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
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

.feature-card {
    background: #F8FAFC;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #E2E8F0;
    min-height: 150px;
}

.feature-title {
    font-size: 19px;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 8px;
}

.feature-text {
    color: #475569;
    font-size: 14px;
}

.metric-card {
    background: #F8FAFC;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #E2E8F0;
}

.section-card {
    background: #FFFFFF;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #E2E8F0;
    margin-bottom: 20px;
}

.small-text {
    color: #64748B;
    font-size: 13px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# DATABASE
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
# USER FUNCTIONS
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
                username.strip(),
                email.strip().lower(),
                hashed_password
            )
        )

        conn.commit()

        return True, "Account created successfully."

    except sqlite3.IntegrityError:

        return False, "Username or email already exists."

    except Exception as e:

        return False, f"Registration error: {e}"


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
        (username.strip(),)
    )

    user = cursor.fetchone()

    if user is None:
        return None

    try:

        if bcrypt.checkpw(
            password.encode("utf-8"),
            user[3]
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

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    for column in df.columns:

        if df[column].dtype == "object":

            df[column] = (
                df[column]
                .astype(str)
                .str.strip()
                .str.lower()
            )

    # THAL
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

    # SEX
    if "sex" in df.columns:

        df["sex"] = df["sex"].replace(
            {
                "female": 0,
                "male": 1
            }
        )

    # FBS
    if "fbs" in df.columns:

        df["fbs"] = df["fbs"].replace(
            {
                "no": 0,
                "yes": 1
            }
        )

    # EXANG
    if "exang" in df.columns:

        df["exang"] = df["exang"].replace(
            {
                "no": 0,
                "yes": 1
            }
        )

    for column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna()

    return df.astype(float)


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

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        results[name] = {

            "model": model,

            "accuracy": accuracy_score(
                y_test,
                predictions
            ),

            "precision": precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "recall": recall_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "f1": f1_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "auc": roc_auc_score(
                y_test,
                probabilities
            ),

            "y_test": y_test,

            "predictions": predictions,

            "probabilities": probabilities
        }

    return results, X.columns.tolist()


# ============================================================
# LOAD DATA
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
        "Missing columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# TRAIN
# ============================================================

try:

    all_results, feature_names = train_models(df)

except Exception as e:

    st.error(
        f"Error training models: {e}"
    )

    st.stop()


best_name = max(
    all_results,
    key=lambda name: all_results[name]["accuracy"]
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_bmi(weight, height):

    if height <= 0:
        return 0

    height_m = height / 100

    return weight / (height_m ** 2)


def bmi_category(bmi):

    if bmi < 18.5:
        return "Underweight"

    elif bmi < 25:
        return "Normal"

    elif bmi < 30:
        return "Overweight"

    else:
        return "Obesity"


def create_excel_file(dataframe):

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        dataframe.to_excel(
            writer,
            index=False,
            sheet_name="Assessments"
        )

    return output.getvalue()


def create_pdf_report(
    username,
    age,
    sex,
    chol,
    trestbps,
    prediction_label,
    risk_level,
    probability,
    model_name,
    accuracy,
    bmi=None
):

    if not REPORTLAB_AVAILABLE:
        return None

    output = io.BytesIO()

    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=10
    )

    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontSize=10,
        leading=15
    )

    elements = []

    elements.append(
        Paragraph(
            "Heart Disease Risk Assessment",
            title_style
        )
    )

    elements.append(
        Paragraph(
            "AI-Powered Cardiovascular Risk Assessment",
            body_style
        )
    )

    elements.append(
        Spacer(1, 15)
    )

    elements.append(
        Paragraph(
            "Patient Information",
            heading_style
        )
    )

    patient_data = [
        ["Username", username],
        ["Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Age", str(age)],
        ["Sex", sex],
        ["Cholesterol", f"{chol} mg/dl"],
        ["Blood Pressure", f"{trestbps} mm Hg"]
    ]

    if bmi is not None:
        patient_data.append(
            ["BMI", f"{bmi:.1f}"]
        )

    table = Table(
        patient_data,
        colWidths=[150, 300]
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E2E8F0")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 8)
            ]
        )
    )

    elements.append(table)

    elements.append(
        Paragraph(
            "Prediction Results",
            heading_style
        )
    )

    prediction_data = [
        ["Prediction", prediction_label],
        ["Risk Level", risk_level],
        ["Risk Probability", f"{probability:.2%}"],
        ["Model", model_name],
        ["Model Accuracy", f"{accuracy:.2%}"]
    ]

    result_table = Table(
        prediction_data,
        colWidths=[150, 300]
    )

    result_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#DBEAFE")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 8)
            ]
        )
    )

    elements.append(result_table)

    elements.append(
        Paragraph(
            "Important Disclaimer",
            heading_style
        )
    )

    elements.append(
        Paragraph(
            "This application provides an educational machine-learning "
            "risk estimate. It is not a medical diagnosis and should not "
            "replace professional medical advice.",
            body_style
        )
    )

    document.build(elements)

    return output.getvalue()


# ============================================================
# LANDING PAGE
# ============================================================

if not st.session_state.logged_in:

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

    # Features

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
<div class="feature-card">

<div class="feature-title">
🤖 AI Prediction
</div>

<div class="feature-text">
Use multiple machine-learning models to estimate cardiovascular risk.
</div>

</div>
""",
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
<div class="feature-card">

<div class="feature-title">
📊 Risk Analysis
</div>

<div class="feature-text">
Understand risk probability, important factors and health alerts.
</div>

</div>
""",
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
<div class="feature-card">

<div class="feature-title">
📈 Track Progress
</div>

<div class="feature-text">
Save assessments and monitor your risk history over time.
</div>

</div>
""",
            unsafe_allow_html=True
        )

    st.write("")

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

        st.subheader(
            "Welcome Back"
        )

        st.write(
            "Sign in to access your health assessments and prediction history."
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
                        message
                    )

                    st.info(
                        "You can now sign in."
                    )

                else:

                    st.error(
                        message
                    )

    st.stop()


# ============================================================
# LOGGED-IN APPLICATION
# ============================================================

st.sidebar.title(
    "❤️ Heart Disease Predictor"
)

st.sidebar.success(
    f"Welcome, {st.session_state.username}"
)


# ============================================================
# SIDEBAR MENU
# ============================================================

menu = st.sidebar.radio(
    "Dashboard",
    [
        "🏠 Dashboard",
        "🩺 New Assessment",
        "📈 Risk History",
        "🧠 Model Performance",
        "🔬 Dataset Analytics",
        "👤 My Profile",
        "🚪 Logout"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if menu == "🏠 Dashboard":

    st.title(
        "Dashboard"
    )

    st.write(
        "Your cardiovascular risk assessment overview."
    )

    records_df = pd.read_sql_query(
        """
        SELECT *
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
            "You haven't completed an assessment yet."
        )

        st.button(
            "Start Your First Assessment",
            type="primary"
        )

    else:

        total = len(records_df)

        average_probability = (
            records_df["probability"].mean()
        )

        latest = records_df.iloc[0]

        high = len(
            records_df[
                records_df["prediction"] == "High Risk"
            ]
        )

        moderate = len(
            records_df[
                records_df["prediction"] == "Moderate Risk"
            ]
        )

        low = len(
            records_df[
                records_df["prediction"] == "Low Risk"
            ]
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Assessments",
                total
            )

        with col2:

            st.metric(
                "Average Risk",
                f"{average_probability:.1%}"
            )

        with col3:

            st.metric(
                "Latest Risk",
                latest["prediction"]
            )

        with col4:

            st.metric(
                "High Risk",
                high
            )

        st.divider()

        st.subheader(
            "Risk Distribution"
        )

        chart_data = pd.DataFrame(
            {
                "Risk Level": [
                    "Low Risk",
                    "Moderate Risk",
                    "High Risk"
                ],
                "Assessments": [
                    low,
                    moderate,
                    high
                ]
            }
        )

        st.bar_chart(
            chart_data.set_index("Risk Level")
        )

        st.subheader(
            "Risk History"
        )

        history = records_df.copy()

        history["prediction_date"] = pd.to_datetime(
            history["prediction_date"]
        )

        history = history.sort_values(
            "prediction_date"
        )

        st.line_chart(
            history.set_index(
                "prediction_date"
            )["probability"]
        )


# ============================================================
# NEW ASSESSMENT
# ============================================================

elif menu == "🩺 New Assessment":

    st.title(
        "New Risk Assessment"
    )

    st.info(
        "This application provides an educational AI-generated risk estimate "
        "and should not replace professional medical advice."
    )

    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------

    selected_model_name = st.selectbox(
        "Choose Prediction Model",
        list(all_results.keys()),
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

    st.caption(
        f"Selected model accuracy: {accuracy:.1%}"
    )

    # --------------------------------------------------------
    # PATIENT INFORMATION
    # --------------------------------------------------------

    st.header(
        "Patient Information"
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50
        )

        sex = st.selectbox(
            "Sex",
            [0, 1],
            format_func=lambda x:
            "Female" if x == 0 else "Male"
        )

        cp = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3],
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
            [0, 1],
            format_func=lambda x:
            "No" if x == 0 else "Yes"
        )

        restecg = st.selectbox(
            "Resting ECG Results",
            [0, 1, 2],
            format_func=lambda x: [
                "Normal",
                "ST-T Abnormality",
                "Left Ventricular Hypertrophy"
            ][x]
        )

    with col2:

        thalach = st.number_input(
            "Max Heart Rate Achieved",
            min_value=60,
            max_value=250,
            value=150
        )

        exang = st.selectbox(
            "Exercise Induced Angina",
            [0, 1],
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
            [0, 1, 2],
            format_func=lambda x: [
                "Upsloping",
                "Flat",
                "Downsloping"
            ][x]
        )

        ca = st.selectbox(
            "Number of Major Vessels (0-3)",
            [0, 1, 2, 3]
        )

        thal = st.selectbox(
            "Thalassemia",
            [0, 1, 2, 3],
            format_func=lambda x: [
                "Normal",
                "Fixed Defect",
                "Reversible Defect",
                "Unknown"
            ][x]
        )

    # --------------------------------------------------------
    # OPTIONAL BMI
    # --------------------------------------------------------

    st.subheader(
        "Optional BMI Information"
    )

    bmi_col1, bmi_col2 = st.columns(2)

    with bmi_col1:

        height = st.number_input(
            "Height (cm)",
            min_value=50.0,
            max_value=250.0,
            value=170.0
        )

    with bmi_col2:

        weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=300.0,
            value=70.0
        )

    bmi = calculate_bmi(
        weight,
        height
    )

    st.metric(
        "BMI",
        f"{bmi:.1f}"
    )

    st.caption(
        f"BMI Category: {bmi_category(bmi)}"
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "Predict Risk",
        type="primary",
        use_container_width=True
    ):

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

        prediction = model.predict(
            input_data
        )[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

        prediction_label = (
            "Positive"
            if prediction == 1
            else "Negative"
        )

        if probability < 0.30:

            risk_level = "Low Risk"

        elif probability < 0.70:

            risk_level = "Moderate Risk"

        else:

            risk_level = "High Risk"

        # ----------------------------------------------------
        # SAVE
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

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        st.divider()

        st.header(
            "Prediction Results"
        )

        if risk_level == "Low Risk":

            st.success(
                f"Low Risk — {probability:.1%} estimated probability"
            )

        elif risk_level == "Moderate Risk":

            st.warning(
                f"Moderate Risk — {probability:.1%} estimated probability"
            )

        else:

            st.error(
                f"High Risk — {probability:.1%} estimated probability"
            )

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

        st.progress(
            int(probability * 100)
        )

        # ----------------------------------------------------
        # HEALTH ALERTS
        # ----------------------------------------------------

        st.subheader(
            "Health Alerts"
        )

        alerts_found = False

        if trestbps > 140:

            st.warning(
                "Elevated resting blood pressure detected."
            )

            alerts_found = True

        if chol > 240:

            st.warning(
                "Elevated cholesterol detected."
            )

            alerts_found = True

        if exang == 1:

            st.warning(
                "Exercise-induced angina was reported."
            )

            alerts_found = True

        if oldpeak > 2:

            st.warning(
                "Elevated ST depression detected."
            )

            alerts_found = True

        if not alerts_found:

            st.success(
                "No additional input-based alerts detected."
            )

        # ----------------------------------------------------
        # RECOMMENDATIONS
        # ----------------------------------------------------

        st.subheader(
            "Educational Recommendations"
        )

        if probability < 0.30:

            st.info(
                """
Continue maintaining healthy lifestyle habits.

Regular physical activity, balanced nutrition,
routine check-ups and monitoring of cardiovascular
risk factors are recommended.
"""
            )

        elif probability < 0.70:

            st.warning(
                """
Consider discussing your cardiovascular risk
factors with a qualified healthcare professional.

Maintaining healthy physical activity, nutrition,
blood pressure and cholesterol levels may be beneficial.
"""
            )

        else:

            st.error(
                """
Consider seeking professional medical advice regarding
your cardiovascular risk factors.

This application cannot diagnose heart disease.
"""
            )

        # ----------------------------------------------------
        # TOP RISK FACTORS
        # ----------------------------------------------------

        st.subheader(
            "Top Risk Factors"
        )

        if hasattr(
            model,
            "feature_importances_"
        ):

            importance_values = (
                model.feature_importances_
            )

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
            "Importance",
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

        st.pyplot(fig)

        plt.close(fig)

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        if SHAP_AVAILABLE and selected_model_name == "Random Forest":

            st.subheader(
                "Explainable AI"
            )

            st.write(
                "SHAP provides an additional explanation of how model features influence predictions."
            )

            try:

                explainer = shap.TreeExplainer(
                    model
                )

                shap_values = explainer.shap_values(
                    input_data
                )

                if isinstance(shap_values, list):

                    shap_values_to_plot = shap_values[1]

                else:

                    shap_values_to_plot = shap_values

                fig_shap, ax_shap = plt.subplots(
                    figsize=(8, 4)
                )

                values = np.abs(
                    shap_values_to_plot[0]
                )

                shap_df = pd.DataFrame(
                    {
                        "Feature": feature_names,
                        "SHAP": values
                    }
                ).sort_values(
                    "SHAP",
                    ascending=False
                ).head(10)

                ax_shap.barh(
                    shap_df["Feature"],
                    shap_df["SHAP"]
                )

                ax_shap.set_title(
                    "SHAP Feature Impact"
                )

                ax_shap.invert_yaxis()

                st.pyplot(
                    fig_shap
                )

                plt.close(fig_shap)

            except Exception:

                st.info(
                    "SHAP explanation could not be generated for this assessment."
                )

        # ----------------------------------------------------
        # PDF REPORT
        # ----------------------------------------------------

        st.subheader(
            "Download Report"
        )

        if REPORTLAB_AVAILABLE:

            pdf_data = create_pdf_report(
                st.session_state.username,
                age,
                "Female" if sex == 0 else "Male",
                chol,
                trestbps,
                prediction_label,
                risk_level,
                probability,
                selected_model_name,
                accuracy,
                bmi
            )

            st.download_button(
                "Download Professional PDF Report",
                data=pdf_data,
                file_name="heart_disease_risk_report.pdf",
                mime="application/pdf"
            )

        else:

            st.warning(
                "ReportLab is not installed. Add reportlab to requirements.txt."
            )


# ============================================================
# RISK HISTORY
# ============================================================

elif menu == "📈 Risk History":

    st.title(
        "Risk History"
    )

    records_df = pd.read_sql_query(
        """
        SELECT
            id AS ID,
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
            "No assessment records available."
        )

    else:

        records_df["Date"] = pd.to_datetime(
            records_df["Date"]
        )

        records_df["Probability_Percent"] = (
            records_df["Probability"] * 100
        ).round(1)

        # ----------------------------------------------------
        # FILTERS
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            risk_filter = st.multiselect(
                "Filter Risk Level",
                [
                    "Low Risk",
                    "Moderate Risk",
                    "High Risk"
                ],
                default=[
                    "Low Risk",
                    "Moderate Risk",
                    "High Risk"
                ]
            )

        with col2:

            model_filter = st.multiselect(
                "Filter Model",
                records_df["Model"].unique(),
                default=list(
                    records_df["Model"].unique()
                )
            )

        filtered_df = records_df[
            records_df["Risk_Level"].isin(
                risk_filter
            )
            &
            records_df["Model"].isin(
                model_filter
            )
        ]

        # ----------------------------------------------------
        # TABLE
        # ----------------------------------------------------

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # TREND
        # ----------------------------------------------------

        st.subheader(
            "Risk Probability Trend"
        )

        trend_df = filtered_df.sort_values(
            "Date"
        )

        if not trend_df.empty:

            st.line_chart(
                trend_df.set_index(
                    "Date"
                )["Probability"]
            )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "Assessment Summary"
        )

        total = len(records_df)

        low = len(
            records_df[
                records_df["Risk_Level"] == "Low Risk"
            ]
        )

        moderate = len(
            records_df[
                records_df["Risk_Level"] == "Moderate Risk"
            ]
        )

        high = len(
            records_df[
                records_df["Risk_Level"] == "High Risk"
            ]
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Total", total)

        with c2:
            st.metric("Low Risk", low)

        with c3:
            st.metric("Moderate Risk", moderate)

        with c4:
            st.metric("High Risk", high)

        # ----------------------------------------------------
        # EXPORT
        # ----------------------------------------------------

        st.subheader(
            "Export Records"
        )

        csv_data = filtered_df.to_csv(
            index=False
        )

        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name="heart_disease_records.csv",
            mime="text/csv"
        )

        excel_data = create_excel_file(
            filtered_df
        )

        st.download_button(
            "Download Excel",
            data=excel_data,
            file_name="heart_disease_records.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # ----------------------------------------------------
        # DELETE RECORD
        # ----------------------------------------------------

        st.subheader(
            "Delete Assessment"
        )

        record_ids = records_df["ID"].tolist()

        selected_id = st.selectbox(
            "Select assessment to delete",
            record_ids
        )

        if st.button(
            "Delete Selected Assessment"
        ):

            cursor.execute(
                """
                DELETE FROM predictions
                WHERE id = ?
                AND user_id = ?
                """,
                (
                    selected_id,
                    st.session_state.user_id
                )
            )

            conn.commit()

            st.success(
                "Assessment deleted."
            )

            st.rerun()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif menu == "🧠 Model Performance":

    st.title(
        "Machine Learning Model Performance"
    )

    st.write(
        "Compare the performance of the three trained classification models."
    )

    performance_df = pd.DataFrame(
        [
            {
                "Model": name,
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1 Score": result["f1"],
                "ROC-AUC": result["auc"]
            }

            for name, result in all_results.items()
        ]
    )

    st.dataframe(
        performance_df.style.format(
            {
                "Accuracy": "{:.1%}",
                "Precision": "{:.1%}",
                "Recall": "{:.1%}",
                "F1 Score": "{:.1%}",
                "ROC-AUC": "{:.3f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.success(
        f"Best accuracy: {best_name}"
    )

    # --------------------------------------------------------
    # ROC CURVES
    # --------------------------------------------------------

    st.subheader(
        "ROC Curve Comparison"
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for name, result in all_results.items():

        fpr, tpr, _ = roc_curve(
            result["y_test"],
            result["probabilities"]
        )

        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {result['auc']:.3f})"
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC-AUC Comparison"
    )

    ax.legend()

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader(
        "Confusion Matrix"
    )

    selected_matrix_model = st.selectbox(
        "Select Model",
        list(all_results.keys()),
        key="confusion_model"
    )

    matrix = confusion_matrix(
        all_results[selected_matrix_model]["y_test"],
        all_results[selected_matrix_model]["predictions"]
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            "Actual Negative",
            "Actual Positive"
        ],
        columns=[
            "Predicted Negative",
            "Predicted Positive"
        ]
    )

    st.dataframe(
        matrix_df,
        use_container_width=True
    )


# ============================================================
# DATASET ANALYTICS
# ============================================================

elif menu == "🔬 Dataset Analytics":

    st.title(
        "Dataset Analytics"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Patient Records",
            len(df)
        )

    with c2:

        st.metric(
            "Features",
            len(feature_names)
        )

    with c3:

        positive_rate = (
            df["target"].mean()
        )

        st.metric(
            "Positive Cases",
            f"{positive_rate:.1%}"
        )

    st.divider()

    # --------------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Target Distribution"
    )

    target_counts = df["target"].value_counts()

    target_df = pd.DataFrame(
        {
            "Target": [
                "Negative",
                "Positive"
            ],
            "Count": [
                int(target_counts.get(0, 0)),
                int(target_counts.get(1, 0))
            ]
        }
    )

    st.bar_chart(
        target_df.set_index("Target")
    )

    # --------------------------------------------------------
    # AGE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Age Distribution"
    )

    age_distribution = pd.DataFrame(
        {
            "Age": df["age"]
        }
    )

    st.bar_chart(
        age_distribution["Age"].value_counts().sort_index()
    )

    # --------------------------------------------------------
    # CHOLESTEROL
    # --------------------------------------------------------

    st.subheader(
        "Cholesterol Statistics"
    )

    chol_col1, chol_col2, chol_col3 = st.columns(3)

    with chol_col1:

        st.metric(
            "Average",
            f"{df['chol'].mean():.1f}"
        )

    with chol_col2:

        st.metric(
            "Minimum",
            f"{df['chol'].min():.0f}"
        )

    with chol_col3:

        st.metric(
            "Maximum",
            f"{df['chol'].max():.0f}"
        )

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PROFILE
# ============================================================

elif menu == "👤 My Profile":

    st.title(
        "My Profile"
    )

    cursor.execute(
        """
        SELECT
            username,
            email
        FROM users
        WHERE id = ?
        """,
        (
            st.session_state.user_id,
        )
    )

    user = cursor.fetchone()

    if user:

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Account Information"
            )

            st.write(
                f"**Username:** {user[0]}"
            )

            st.write(
                f"**Email:** {user[1]}"
            )

        with col2:

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM predictions
                WHERE user_id = ?
                """,
                (
                    st.session_state.user_id,
                )
            )

            assessment_count = cursor.fetchone()[0]

            st.metric(
                "Total Assessments",
                assessment_count
            )

    st.divider()

    # --------------------------------------------------------
    # PASSWORD CHANGE
    # --------------------------------------------------------

    st.subheader(
        "Change Password"
    )

    current_password = st.text_input(
        "Current Password",
        type="password"
    )

    new_password = st.text_input(
        "New Password",
        type="password"
    )

    confirm_new_password = st.text_input(
        "Confirm New Password",
        type="password"
    )

    if st.button(
        "Change Password"
    ):

        cursor.execute(
            """
            SELECT password
            FROM users
            WHERE id = ?
            """,
            (
                st.session_state.user_id,
            )
        )

        stored_password = cursor.fetchone()[0]

        if not bcrypt.checkpw(
            current_password.encode("utf-8"),
            stored_password
        ):

            st.error(
                "Current password is incorrect."
            )

        elif len(new_password) < 6:

            st.error(
                "New password must contain at least 6 characters."
            )

        elif new_password != confirm_new_password:

            st.error(
                "New passwords do not match."
            )

        else:

            new_hash = bcrypt.hashpw(
                new_password.encode("utf-8"),
                bcrypt.gensalt()
            )

            cursor.execute(
                """
                UPDATE users
                SET password = ?
                WHERE id = ?
                """,
                (
                    new_hash,
                    st.session_state.user_id
                )
            )

            conn.commit()

            st.success(
                "Password changed successfully."
            )


# ============================================================
# LOGOUT
# ============================================================

elif menu == "🚪 Logout":

    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None

    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Heart Disease Risk Predictor"
)

st.sidebar.caption(
    "Educational machine-learning application."
)

st.sidebar.caption(
    "Not a medical diagnosis."
)
