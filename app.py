
import streamlit as st
import tensorflow as tf
import numpy as np
import hmac
from PIL import Image

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Cotton Seed Quality Detection",
    page_icon="🌱",
    layout="wide"
)

# ==================================================
# 2. GREEN AND WHITE THEME
# ==================================================
st.markdown("""
<style>
.stApp {
    background-color: #FFFFFF;
    color: #183D20;
}

[data-testid="stSidebar"] {
    background-color: #EDF7ED;
}

h1, h2, h3 {
    color: #218838 !important;
}

div.stButton > button {
    background-color: #218838;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #176B2C;
    color: white;
}

div[data-testid="stFormSubmitButton"] button {
    background-color: #218838;
    color: white;
    border-radius: 8px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# 3. SESSION STATE
# ==================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "role" not in st.session_state:
    st.session_state.role = ""

if "history" not in st.session_state:
    st.session_state.history = []

# ==================================================
# 4. LOAD TRAINED MODEL
# ==================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "cotton_seed_mobilenetv3.keras"
    )

# ==================================================
# 5. AUTHENTICATION
# ==================================================
def authenticate(username, password):
    users = st.secrets.get("users", {})

    for saved_username, details in users.items():
        if not isinstance(details, dict):
            continue

        saved_password = str(details.get("password", ""))
        saved_role = str(details.get("role", "user")).lower()

        if (
            hmac.compare_digest(username, str(saved_username))
            and hmac.compare_digest(password, saved_password)
        ):
            return True, str(saved_username), saved_role

    return False, "", ""


# ==================================================
# 6. LOGIN PAGE
# ==================================================
if not st.session_state.logged_in:
    st.title("🌱 Cotton Seed Quality Detection")
    st.subheader("AI-Based Cotton Seed Quality Analysis")
    st.divider()

    st.subheader("Login to Your Account")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        submitted = st.form_submit_button(
            "Login",
            use_container_width=True
        )

    if submitted:
        if not username.strip() or not password:
            st.error("Please enter your username and password.")
        else:
            success, saved_username, saved_role = authenticate(
                username.strip(),
                password
            )

            if success:
                st.session_state.logged_in = True

                # Use the configured account username
                # throughout the application.
                st.session_state.username = saved_username
                st.session_state.role = saved_role
                st.session_state.history = []

                st.rerun()
            else:
                st.error("Invalid username or password.")

    st.stop()


# ==================================================
# 7. APPLICATION HEADER
# ==================================================
st.title("🌱 Cotton Seed Quality Detection")
st.caption("AI-Powered Cotton Seed Quality Analysis")
st.divider()


# ==================================================
# 8. SIDEBAR AND LOGOUT
# ==================================================
with st.sidebar:
    st.title("🌱 Navigation")

    st.write("**Logged-in Username:**")
    st.success(st.session_state.username)

    st.write("**Role:**")
    st.info(st.session_state.role.title())

    if st.session_state.role == "admin":
        page = st.radio(
            "Choose a page",
            [
                "Logged-in User Dashboard",
                "Admin Dashboard",
                "Seed Prediction"
            ]
        )
    else:
        page = st.radio(
            "Choose a page",
            [
                "Logged-in User Dashboard",
                "Seed Prediction"
            ]
        )

    st.divider()

    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.role = ""
        st.session_state.history = []
        st.rerun()


# ==================================================
# 9. LOGGED-IN USER DASHBOARD
# ==================================================
if page == "Logged-in User Dashboard":
    st.title("👤 Logged-in User Dashboard")

    # Display the exact username used to log in.
    st.subheader(
        f"Welcome, {st.session_state.username}!"
    )

    st.write("Your account information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Username",
            st.session_state.username
        )

    with col2:
        st.metric(
            "Account Role",
            st.session_state.role.title()
        )

    st.divider()
    st.subheader("📊 Your Prediction Statistics")

    history = st.session_state.history

    total = len(history)

    good_count = sum(
        item["Quality"] == "Good" for item in history
    )
    broken_count = sum(
        item["Quality"] == "Broken" for item in history
    )
    cracked_count = sum(
        item["Quality"] == "Cracked" for item in history
    )

    col1, col2 = st.columns(2)
    col1.metric("Total Predictions", total)
    col2.metric("Good Seeds", good_count)

    col3, col4 = st.columns(2)
    col3.metric("Broken Seeds", broken_count)
    col4.metric("Cracked Seeds", cracked_count)

    st.divider()
    st.subheader("📋 Your Prediction History")

    if history:
        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

        if st.button("Clear My Prediction History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info(
            "No predictions yet. Go to Seed Prediction "
            "to capture or upload a cotton seed image."
        )


# ==================================================
# 10. ADMIN DASHBOARD
# ==================================================
elif page == "Admin Dashboard":
    st.title("🛡️ Admin Dashboard")

    st.subheader(
        f"Welcome, Administrator {st.session_state.username}!"
    )

    history = st.session_state.history

    total = len(history)
    good_count = sum(
        item["Quality"] == "Good" for item in history
    )
    broken_count = sum(
        item["Quality"] == "Broken" for item in history
    )
    cracked_count = sum(
        item["Quality"] == "Cracked" for item in history
    )

    st.subheader("Current Session Statistics")

    col1, col2 = st.columns(2)
    col1.metric("Total Predictions", total)
    col2.metric("Good Seeds", good_count)

    col3, col4 = st.columns(2)
    col3.metric("Broken Seeds", broken_count)
    col4.metric("Cracked Seeds", cracked_count)

    st.divider()
    st.subheader("Prediction Records")

    if history:
        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

        if st.button("Clear Current Session History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions recorded in this session.")

    st.caption(
        "These records are temporary and session-based."
    )


# ==================================================
# 11. SEED PREDICTION
# ==================================================
elif page == "Seed Prediction":
    st.title("🔬 Cotton Seed Prediction")

    st.write(
        "Capture a new photo or upload an existing "
        "cotton seed image."
    )

    input_method = st.radio(
        "Choose image input method:",
        ["Capture Photo", "Upload Existing Image"],
        horizontal=True
    )

    image_file = None

    if input_method == "Capture Photo":
        image_file = st.camera_input(
            "Take a photo of the cotton seeds"
        )
    else:
        image_file = st.file_uploader(
            "Choose a cotton seed image",
            type=["jpg", "jpeg", "png"]
        )

    if image_file is not None:
        try:
            image = Image.open(image_file).convert("RGB")

            st.image(
                image,
                caption="Selected Cotton Seed Image",
                use_container_width=True
            )

            if st.button(
                "Predict Seed Quality",
                use_container_width=True
            ):
                with st.spinner("Analyzing seed image..."):
                    model = load_model()

                    resized_image = image.resize((224, 224))
                    image_array = np.array(
                        resized_image,
                        dtype=np.float32
                    )
                    image_array = np.expand_dims(
                        image_array,
                        axis=0
                    )

                    predictions = model.predict(
                        image_array,
                        verbose=0
                    )[0]

                    # Must match the class order
                    # used during model training.
                    classes = ["Broken", "Cracked", "Good"]

                    predicted_index = int(
                        np.argmax(predictions)
                    )
                    predicted_class = classes[predicted_index]
                    confidence = float(
                        predictions[predicted_index] * 100
                    )

                record = {
                    "Username": st.session_state.username,
                    "Role": st.session_state.role.title(),
                    "Quality": predicted_class,
                    "Confidence": f"{confidence:.2f}%"
                }

                st.session_state.history.append(record)

                st.success("Prediction completed!")

                if predicted_class == "Good":
                    st.success(
                        f"Predicted Quality: {predicted_class}"
                    )
                else:
                    st.warning(
                        f"Predicted Quality: {predicted_class}"
                    )

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

                st.subheader("Class Confidence Scores")

                for index, class_name in enumerate(classes):
                    score = float(predictions[index] * 100)

                    st.write(
                        f"**{class_name}: {score:.2f}%**"
                    )
                    st.progress(
                        min(max(score / 100, 0.0), 1.0)
                    )

                st.caption(
                    "These are AI model estimates. Verify results "
                    "before making commercial seed-quality decisions."
                )

        except Exception:
            st.error(
                "Unable to process the image or load the model. "
                "Check the model file and app dependencies."
            )

    st.divider()
    st.subheader("Recent Predictions")

    if st.session_state.history:
        st.dataframe(
            st.session_state.history,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Your predictions will appear here.")
