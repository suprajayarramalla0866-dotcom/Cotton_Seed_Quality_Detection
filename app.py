
import hmac
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Cotton Seed Quality Detection",
    page_icon="🌱",
    layout="wide",
)

# ---------------- DESIGN ----------------
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
    div.stButton > button {
        background-color: #198754;
        color: white;
        border-radius: 8px;
        border: none;
        min-height: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = "cotton_seed_mobilenetv3.keras"
IMAGE_SIZE = (224, 224)

# Must match the class order used when training your model.
CLASS_NAMES = ["Broken", "Cracked", "Good"]


# ---------------- SESSION STATE ----------------
defaults = {
    "logged_in": False,
    "username": "",
    "role": "",
    "history": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


# ---------------- LOGIN ----------------
def check_login(username, password):
    try:
        users = st.secrets["users"]

        for saved_username, details in users.items():
            saved_password = str(details["password"])

            if (
                hmac.compare_digest(username, str(saved_username))
                and hmac.compare_digest(password, saved_password)
            ):
                return str(saved_username), str(
                    details.get("role", "user")
                ).lower()

    except (KeyError, TypeError):
        return None

    return None


def login_page():
    st.title("🌱 Cotton Seed Quality Detection")
    st.write("AI-based cotton seed quality classification")
    st.divider()

    _, center, _ = st.columns([1, 1.4, 1])

    with center:
        st.subheader("Login to Your Account")

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(
                "Login",
                use_container_width=True,
            )

        if submitted:
            if not username.strip() or not password:
                st.warning("Enter both username and password.")
            else:
                result = check_login(username.strip(), password)

                if result:
                    st.session_state.username = result[0]
                    st.session_state.role = result[1]
                    st.session_state.logged_in = True
                    st.session_state.history = []
                    st.rerun()
                else:
                    st.error(
                        "Login failed. Check your username, password, "
                        "and Streamlit Secrets."
                    )


# ---------------- PREDICTION ----------------
def predict_image(image):
    model = load_model()

    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    # Assumes the model was trained on RGB pixel values in the 0–255 range.
    output = np.asarray(model.predict(array, verbose=0))[0]

    if len(output) != len(CLASS_NAMES):
        raise ValueError(
            "Model output count does not match CLASS_NAMES."
        )

    index = int(np.argmax(output))
    label = CLASS_NAMES[index]

    # Appropriate for a softmax output model.
    confidence = float(output[index]) * 100

    scores = {
        CLASS_NAMES[i]: float(output[i]) * 100
        for i in range(len(CLASS_NAMES))
    }

    return label, confidence, scores


def prediction_page():
    st.header("🔬 Seed Quality Prediction")
    st.write("Capture a photo or upload a cotton seed image.")

    method = st.radio(
        "Select image source",
        ["Capture Photo", "Upload Image"],
        horizontal=True,
    )

    uploaded_image = None

    if method == "Capture Photo":
        uploaded_image = st.camera_input("Capture cotton seeds")
    else:
        uploaded_image = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
        )

    if uploaded_image is None:
        return

    try:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Selected Image", width=350)
    except Exception:
        st.error("The selected file is not a valid image.")
        return

    if st.button("Predict Seed Quality", use_container_width=True):
        try:
            with st.spinner("Analysing the image..."):
                label, confidence, scores = predict_image(image)

            st.subheader("Prediction Result")

            if label == "Good":
                st.success(f"Quality: {label}")
            else:
                st.warning(f"Quality: {label}")

            st.metric("Confidence", f"{confidence:.2f}%")

            st.subheader("Class Scores")
            for name, score in scores.items():
                st.write(f"**{name}: {score:.2f}%**")
                st.progress(min(max(score / 100, 0.0), 1.0))

            st.session_state.history.append({
                "Username": st.session_state.username,
                "Quality": label,
                "Confidence": f"{confidence:.2f}%",
                "Date and Time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            })

        except Exception as error:
            st.error("Prediction could not be completed.")
            st.exception(error)


# ---------------- USER DASHBOARD ----------------
def user_dashboard():
    username = st.session_state.username
    history = st.session_state.history

    st.title(f"👤 {username}'s Dashboard")
    st.subheader(f"Welcome, {username}!")
    st.write("View your account details and prediction statistics.")

    first, second = st.columns(2)
    first.metric("Username", username)
    second.metric("Role", st.session_state.role.title())

    st.divider()
    st.subheader("Prediction Statistics")

    good = sum(row["Quality"] == "Good" for row in history)
    broken = sum(row["Quality"] == "Broken" for row in history)
    cracked = sum(row["Quality"] == "Cracked" for row in history)

    a, b, c, d = st.columns(4)
    a.metric("Total Predictions", len(history))
    b.metric("Good", good)
    c.metric("Broken", broken)
    d.metric("Cracked", cracked)

    st.subheader("Prediction History")

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
        st.info("No predictions yet. Start from Seed Prediction.")


# ---------------- ADMIN DASHBOARD ----------------
def admin_dashboard():
    st.title("🛠️ Admin Dashboard")
    st.write("Prediction records available in this session.")

    history = st.session_state.history
    st.metric("Total Predictions", len(history))

    if history:
        st.dataframe(
            list(reversed(history)),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No predictions recorded in this session.")


# ---------------- MAIN APP ----------------
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


# ---------------- RUN ----------------
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        st.error("The app encountered an unexpected error.")
        st.exception(error)
