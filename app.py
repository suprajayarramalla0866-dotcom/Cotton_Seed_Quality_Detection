
import streamlit as st
import sqlite3
import hashlib
import secrets
import hmac
import numpy as np

from datetime import datetime
from pathlib import Path
from PIL import Image

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Cotton Seed Quality Detection",
    page_icon="🌱",
    layout="wide"
)

# ---------------- PATHS AND SETTINGS ----------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "cotton_seed_mobilenetv3.keras"
DB_PATH = BASE_DIR / "users.db"

CLASS_NAMES = ["Broken", "Cracked", "Good"]
IMG_SIZE = (224, 224)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #F7FBF7;
}

h1, h2, h3 {
    color: #064D25;
}

/* Dark green heading text only — no box */
.app-header {
    color: #064D25;
    font-size: 2.35rem;
    font-weight: 800;
    text-align: center;
    padding: 10px 0;
    background: transparent;
    border: none;
    box-shadow: none;
}

.stSidebar {
    background-color: #E4F0E5;
}

.stButton > button {
    background-color: #064D25;
    color: white;
    border-radius: 8px;
    border: none;
}

.stButton > button:hover {
    background-color: #0B6B35;
    color: white;
}

[data-testid="stMetric"] {
    background-color: white;
    border-radius: 8px;
    padding: 12px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- DATABASE ----------------
def get_connection():
    return sqlite3.connect(DB_PATH)


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL
            )
        """)


def hash_password(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000
    ).hex()


def create_user(username, password, role="user"):
    username = username.strip()

    if not username or not password:
        return False, "Enter both username and password."

    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)

    try:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO users
                (username, password_hash, salt, role)
                VALUES (?, ?, ?, ?)
                """,
                (username, password_hash, salt, role)
            )
        return True, "Account created successfully."

    except sqlite3.IntegrityError:
        return False, "That username already exists."


def check_login(username, password, selected_role):
    with get_connection() as conn:
        result = conn.execute(
            """
            SELECT password_hash, salt, role
            FROM users
            WHERE username = ?
            """,
            (username.strip(),)
        ).fetchone()

    if result is None:
        return False

    stored_hash, salt, stored_role = result
    entered_hash = hash_password(password, salt)

    return (
        hmac.compare_digest(stored_hash, entered_hash)
        and stored_role.lower() == selected_role.lower()
    )


def initialize_admin():
    """Create the first admin from Streamlit Secrets, if configured."""
    try:
        admin_username = st.secrets["ADMIN_USERNAME"]
        admin_password = st.secrets["ADMIN_PASSWORD"]
    except Exception:
        return

    if not admin_username or not admin_password:
        return

    with get_connection() as conn:
        existing = conn.execute(
            "SELECT username FROM users WHERE username = ?",
            (admin_username.strip(),)
        ).fetchone()

    if existing is None:
        create_user(admin_username, admin_password, "admin")


# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Place the model in the same folder as app.py."
        )

    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)


# ---------------- HEADER ----------------
def show_header():
    st.markdown(
        '<div class="app-header">🌱 Cotton Seed Quality Detection</div>',
        unsafe_allow_html=True
    )


# ---------------- SESSION STATE ----------------
def initialize_session():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if "username" not in st.session_state:
        st.session_state.username = ""

    if "role" not in st.session_state:
        st.session_state.role = ""

    if "history" not in st.session_state:
        st.session_state.history = []


# ---------------- LOGIN / SIGNUP ----------------
def login_page():
    show_header()

    st.caption("AI-based cotton seed quality analysis")
    st.divider()

    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:
        st.subheader("Welcome Back")

        username = st.text_input(
            "Username",
            key="login_username"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        role = st.radio(
            "Login as",
            ["User", "Admin"],
            horizontal=True,
            key="login_role"
        )

        if st.button("Login", use_container_width=True):
            if check_login(username, password, role):
                st.session_state.logged_in = True
                st.session_state.username = username.strip()
                st.session_state.role = role.lower()
                st.rerun()
            else:
                st.error("Invalid username, password, or selected role.")

    with signup_tab:
        st.subheader("Create a User Account")

        new_username = st.text_input(
            "Choose a username",
            key="signup_username"
        )

        new_password = st.text_input(
            "Choose a password",
            type="password",
            key="signup_password"
        )

        confirm_password = st.text_input(
            "Confirm password",
            type="password",
            key="signup_confirm_password"
        )

        if st.button("Create Account", use_container_width=True):
            if len(new_password) < 8:
                st.error("Use a password with at least 8 characters.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = create_user(
                    new_username,
                    new_password,
                    "user"
                )

                if success:
                    st.success(message)
                else:
                    st.error(message)


# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ---------------- PREDICTION ----------------
def predict_seed(image):
    model = load_model()
    processed_image = preprocess_image(image)

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    scores = np.asarray(predictions[0], dtype=float)
    predicted_index = int(np.argmax(scores))

    if predicted_index >= len(CLASS_NAMES):
        raise ValueError(
            "The number of model outputs does not match CLASS_NAMES."
        )

    return CLASS_NAMES[predicted_index], scores


# ---------------- USER DASHBOARD ----------------
def user_dashboard():
    st.subheader(f"👤 {st.session_state.username}'s Dashboard")

    st.write(
        "Upload a cotton seed image or capture one using your camera "
        "to identify its predicted quality category."
    )

    camera_image = st.camera_input(
        "Capture a cotton seed image",
        key="camera_image"
    )

    uploaded_image = st.file_uploader(
        "Or upload an image",
        type=["jpg", "jpeg", "png"],
        key="uploaded_image"
    )

    image_file = uploaded_image if uploaded_image is not None else camera_image

    if image_file is not None:
        try:
            image = Image.open(image_file).convert("RGB")

            st.image(
                image,
                caption="Selected cotton seed image",
                width=350
            )

            if st.button("Predict Seed Quality", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    predicted_class, scores = predict_seed(image)

                confidence = float(scores.max()) * 100

                st.success(f"Predicted quality: {predicted_class}")
                st.metric("Prediction confidence", f"{confidence:.2f}%")

                st.subheader("Class Scores")

                for index, class_name in enumerate(CLASS_NAMES):
                    if index < len(scores):
                        st.write(
                            f"**{class_name}:** "
                            f"{float(scores[index]) * 100:.2f}%"
                        )
                        st.progress(
                            min(max(float(scores[index]), 0.0), 1.0)
                        )

                st.session_state.history.append({
                    "username": st.session_state.username,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "prediction": predicted_class,
                    "confidence": f"{confidence:.2f}%"
                })

        except Exception as error:
            st.error("Unable to process this image.")
            st.exception(error)

    st.divider()
    st.subheader("Your Current Session History")

    user_history = [
        item for item in st.session_state.history
        if item["username"] == st.session_state.username
    ]

    if user_history:
        st.dataframe(
            [
                {
                    "Time": item["time"],
                    "Prediction": item["prediction"],
                    "Confidence": item["confidence"]
                }
                for item in reversed(user_history)
            ],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No predictions have been made in this session yet.")


# ---------------- ADMIN DASHBOARD ----------------
def admin_dashboard():
    st.subheader("🛡️ Admin Dashboard")

    with get_connection() as conn:
        users = conn.execute(
            "SELECT username, role FROM users ORDER BY username"
        ).fetchall()

    col1, col2 = st.columns(2)

    col1.metric("Registered Accounts", len(users))
    col2.metric(
        "Predictions in Current Session",
        len(st.session_state.history)
    )

    st.subheader("Registered Accounts")

    if users:
        st.dataframe(
            [
                {"Username": username, "Role": role.title()}
                for username, role in users
            ],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No registered accounts found.")

    st.subheader("Prediction History — Current Session")

    if st.session_state.history:
        st.dataframe(
            [
                {
                    "Username": item["username"],
                    "Time": item["time"],
                    "Prediction": item["prediction"],
                    "Confidence": item["confidence"]
                }
                for item in reversed(st.session_state.history)
            ],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No predictions have been made in this session.")


# ---------------- MAIN APPLICATION ----------------
def main():
    initialize_database()
    initialize_admin()
    initialize_session()

    if not st.session_state.logged_in:
        login_page()
        return

    with st.sidebar:
        st.markdown("## 🌱 Cotton Seed App")
        st.write(f"**Username:** {st.session_state.username}")
        st.write(f"**Role:** {st.session_state.role.title()}")

        st.divider()

        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.role = ""
            st.rerun()

    show_header()
    st.caption("AI-based cotton seed quality analysis")
    st.divider()

    if st.session_state.role == "admin":
        admin_dashboard()
    else:
        user_dashboard()


# ---------------- RUN APPLICATION ----------------
try:
    main()
except Exception as error:
    st.error("The application encountered an error.")
    st.exception(error)
