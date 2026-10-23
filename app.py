
import streamlit as st
import numpy as np
import sqlite3
import hashlib
import secrets
import hmac
from datetime import datetime
from pathlib import Path
from PIL import Image

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Cotton Seed Quality Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================
# 2. GREEN AND WHITE DESIGN
# ==================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7fbf7;
    }

    h1, h2, h3 {
        color: #176b35;
    }

    [data-testid="stSidebar"] {
        background-color: #e7f4e8;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background-color: #198754;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        min-height: 40px;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background-color: #126b3b;
        color: white;
    }

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #d5e8d8;
        padding: 12px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# 3. APP SETTINGS
# ==================================================
MODEL_PATH = Path(__file__).parent / "cotton_seed_mobilenetv3.keras"
DB_PATH = Path(__file__).parent / "users.db"

IMAGE_SIZE = (224, 224)

# IMPORTANT:
# This order must match your model's training class order.
CLASS_NAMES = ["Broken", "Cracked", "Good"]


# ==================================================
# 4. SESSION STATE
# ==================================================
defaults = {
    "logged_in": False,
    "username": "",
    "role": "",
    "history": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# 5. DATABASE
# ==================================================
def get_connection():
    return sqlite3.connect(str(DB_PATH), timeout=10)


def init_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user'
            )
        """)


def hash_password(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        200000,
    ).hex()


def register_user(username, password):
    username = username.strip()

    if len(username) < 3:
        return False, "Username must have at least 3 characters."

    if len(username) > 40:
        return False, "Username must be 40 characters or fewer."

    if not username.replace("_", "").replace("-", "").isalnum():
        return False, "Use letters, numbers, underscores, or hyphens."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)

    try:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO users
                    (username, password_hash, salt, role)
                VALUES (?, ?, ?, 'user')
                """,
                (username, password_hash, salt),
            )

        return True, "Your account was created successfully."

    except sqlite3.IntegrityError:
        return False, "This username already exists."

    except sqlite3.Error:
        return False, (
            "Account storage is unavailable. Please try again later."
        )


def check_login(username, password):
    try:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT username, password_hash, salt, role
                FROM users
                WHERE username = ? COLLATE NOCASE
                """,
                (username.strip(),),
            ).fetchone()

    except sqlite3.Error:
        return None

    if row is None:
        return None

    saved_username, saved_hash, salt, role = row
    entered_hash = hash_password(password, salt)

    if hmac.compare_digest(entered_hash, saved_hash):
        return saved_username, role

    return None


# Create the database and users table.
init_database()


# ==================================================
# 6. LOAD AI MODEL
# ==================================================
@st.cache_resource(show_spinner=False)
def load_model():
    import tensorflow as tf

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )


# ==================================================
# 7. PREDICT COTTON SEED QUALITY
# ==================================================
def predict_image(image):
    model = load_model()

    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # Assumes the model was trained with RGB pixel values
    # in the 0–255 range, without additional normalization.
    predictions = np.asarray(
        model.predict(image_array, verbose=0)
    )[0]

    if len(predictions) != len(CLASS_NAMES):
        raise ValueError(
            "The number of model outputs does not match CLASS_NAMES."
        )

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(predictions[predicted_index]) * 100

    scores = {
        CLASS_NAMES[i]: float(predictions[i]) * 100
        for i in range(len(CLASS_NAMES))
    }

    return predicted_class, confidence, scores


# ==================================================
# 8. LOGIN AND SIGN-UP PAGE
# ==================================================
def login_page():
    st.title("🌱 Cotton Seed Quality Detection")
    st.caption("AI-based cotton seed quality classification")
    st.divider()

    _, center, _ = st.columns([1, 1.5, 1])

    with center:
        login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

        # ---------------- LOGIN ----------------
        with login_tab:
            st.subheader("Welcome Back")

            with st.form("login_form"):
                username = st.text_input(
                    "Username",
                    key="login_username",
                )

                password = st.text_input(
                    "Password",
                    type="password",
                    key="login_password",
                )

                login_clicked = st.form_submit_button(
                    "Login",
                    use_container_width=True,
                )

            if login_clicked:
                if not username.strip() or not password:
                    st.warning("Enter your username and password.")
                else:
                    result = check_login(username, password)

                    if result:
                        st.session_state.username = result[0]
                        st.session_state.role = result[1]
                        st.session_state.logged_in = True
                        st.session_state.history = []
                        st.rerun()
                    else:
                        st.error(
                            "Login failed. Check your details or "
                            "create an account using Sign Up."
                        )

        # ---------------- SIGN UP ----------------
        with signup_tab:
            st.subheader("Create a New Account")

            with st.form("signup_form"):
                new_username = st.text_input(
                    "Choose a username",
                    key="signup_username",
                )

                new_password = st.text_input(
                    "Create a password",
                    type="password",
                    key="signup_password",
                )

                confirm_password = st.text_input(
                    "Confirm password",
                    type="password",
                    key="signup_confirm_password",
                )

                signup_clicked = st.form_submit_button(
                    "Create Account",
                    use_container_width=True,
                )

            if signup_clicked:
                if (
                    not new_username.strip()
                    or not new_password
                    or not confirm_password
                ):
                    st.warning("Please fill in all fields.")

                elif new_password != confirm_password:
                    st.error("The passwords do not match.")

                else:
                    success, message = register_user(
                        new_username,
                        new_password,
                    )

                    if success:
                        st.success(message)
                        st.info(
                            "Your account is ready. Open the Login "
                            "tab to sign in."
                        )
                    else:
                        st.error(message)


# ==================================================
# 9. USER DASHBOARD
# ==================================================
def user_dashboard():
    username = st.session_state.username
    history = st.session_state.history

    st.title(f"👤 {username}'s Dashboard")
    st.subheader(f"Welcome, {username}!")

    st.write(
        "View your account information and cotton seed "
        "prediction statistics."
    )

    col1, col2 = st.columns(2)
    col1.metric("Username", username)
    col2.metric("Account Role", st.session_state.role.title())

    st.divider()
    st.subheader("📊 Prediction Statistics")

    total = len(history)
    good = sum(item["Quality"] == "Good" for item in history)
    broken = sum(item["Quality"] == "Broken" for item in history)
    cracked = sum(item["Quality"] == "Cracked" for item in history)

    a, b, c, d = st.columns(4)
    a.metric("Total Predictions", total)
    b.metric("Good Seeds", good)
    c.metric("Broken Seeds", broken)
    d.metric("Cracked Seeds", cracked)

    st.divider()
    st.subheader("🕒 Prediction History")

    if history:
        st.dataframe(
            list(reversed(history)),
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Clear My History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info(
            "No predictions yet. Open Seed Prediction to analyse an image."
        )


# ==================================================
# 10. ADMIN DASHBOARD
# ==================================================
def admin_dashboard():
    st.title("🛠️ Admin Dashboard")
    st.write(
        "View prediction records available in the current session."
    )

    history = st.session_state.history

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Predictions", len(history))
    col2.metric(
        "Good Seeds",
        sum(item["Quality"] == "Good" for item in history),
    )
    col3.metric(
        "Other Results",
        sum(item["Quality"] != "Good" for item in history),
    )

    st.divider()
    st.subheader("Prediction Records")

    if history:
        st.dataframe(
            list(reversed(history)),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No prediction records are available in this session.")


# ==================================================
# 11. SEED PREDICTION PAGE
# ==================================================
def prediction_page():
    st.title("🔬 Cotton Seed Quality Prediction")

    st.write(
        "Capture a photo or upload a cotton seed image "
        "to classify it using your trained model."
    )

    method = st.radio(
        "Choose image source",
        ["Capture Photo", "Upload Existing Image"],
        horizontal=True,
    )

    image_file = None

    if method == "Capture Photo":
        image_file = st.camera_input(
            "Take a photo of the cotton seeds"
        )
    else:
        image_file = st.file_uploader(
            "Upload a cotton seed image",
            type=["jpg", "jpeg", "png"],
        )

    if image_file is None:
        return

    try:
        image = Image.open(image_file).convert("RGB")
    except Exception:
        st.error("Unable to open this image. Choose another image.")
        return

    st.image(
        image,
        caption="Selected Cotton Seed Image",
        width=350,
    )

    if st.button(
        "Predict Seed Quality",
        use_container_width=True,
    ):
        try:
            with st.spinner("Loading model and analysing image..."):
                label, confidence, scores = predict_image(image)

            st.divider()
            st.subheader("Prediction Result")

            if label == "Good":
                st.success(f"🌱 Quality: {label}")
            else:
                st.warning(f"Quality: {label}")

            st.metric("Model Confidence", f"{confidence:.2f}%")

            st.subheader("Class Scores")

            for class_name, score in scores.items():
                st.write(f"**{class_name}: {score:.2f}%**")
                st.progress(
                    min(max(score / 100, 0.0), 1.0)
                )

            st.session_state.history.append({
                "Username": st.session_state.username,
                "Role": st.session_state.role.title(),
                "Quality": label,
                "Confidence": f"{confidence:.2f}%",
                "Date and Time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            })

        except Exception as error:
            st.error(
                "Prediction failed. Check the model file and "
                "its preprocessing settings."
            )
            st.exception(error)


# ==================================================
# 12. MAIN APPLICATION
# ==================================================
def main():
    if not st.session_state.logged_in:
        login_page()
        return

    with st.sidebar:
        st.title("🌱 Cotton Seed App")

        st.write("Logged-in Username")
        st.success(st.session_state.username)

        st.write("Account Role")
        st.info(st.session_state.role.title())

        st.divider()

        pages = ["User Dashboard"]

        if st.session_state.role == "admin":
            pages.append("Admin Dashboard")

        pages.append("Seed Prediction")

        page = st.radio("Navigation", pages)

        st.divider()

        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.role = ""
            st.session_state.history = []
            st.rerun()

    st.title("🌱 Cotton Seed Quality Detection")
    st.caption("AI-powered cotton seed image classification")
    st.divider()

    if page == "User Dashboard":
        user_dashboard()

    elif page == "Admin Dashboard":
        admin_dashboard()

    elif page == "Seed Prediction":
        prediction_page()


# ==================================================
# 13. START APPLICATION
# ==================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        st.error("The application encountered an unexpected error.")
        st.exception(error)
