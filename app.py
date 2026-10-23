
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
    layout="centered"
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
    border: 1px solid #176B2C;
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
# 5. USER AUTHENTICATION
# ==================================================
def authenticate(username, password):
    users = st.secrets.get("users", {})

    for saved_username, details in users.items():
        if not isinstance(details, dict):
            continue

        saved_password = str(details.get("password", ""))
        saved_role = str(
            details.get("role", "user")
        ).lower()

        if (
            hmac.compare_digest(
                username, str(saved_username)
            )
            and hmac.compare_digest(
                password, saved_password
            )
        ):
            return True, str(saved_username), saved_role

    return False, "", ""

# ==================================================
# 6. LOGIN PAGE
# ==================================================
if not st.session_state.logged_in:

    st.markdown("""
    <div style="
        background-color:#218838;
        padding:25px;
        border-radius:15px;
        text-align:center;
        color:white;">

        <h1 style="color:white !important;">
            🌱 Cotton Seed Quality Detection
        </h1>

        <p>AI-Based Cotton Seed Quality Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.subheader("Login to Your Account")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input(
            "Password",
            type="password"
        )

        submitted = st.form_submit_button(
            "Login",
            use_container_width=True
        )

    if submitted:
        if not username.strip() or not password:
            st.error(
                "Please enter both username and password."
            )
        else:
            success, saved_username, saved_role = authenticate(
                username.strip(),
                password
            )

            if success:
                st.session_state.logged_in = True
                st.session_state.username = saved_username
                st.session_state.role = saved_role
                st.session_state.history = []
                st.rerun()
            else:
                st.error("Invalid username or password.")

    st.stop()

# ==================================================
# 7. LOGGED-IN HEADER
# ==================================================
st.markdown("""
<div style="
    background-color:#218838;
    padding:18px;
    border-radius:12px;
    text-align:center;
    color:white;">

    <h1 style="color:white !important; margin:0;">
        🌱 Cotton Seed Quality Detection
    </h1>

    <p style="margin:8px 0 0 0;">
        AI-Powered Seed Quality Analysis
    </p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ==================================================
# 8. SIDEBAR NAVIGATION AND LOGOUT
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

    if st.button(
        "Logout",
        use_container_width=True
    ):
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

    st.write(
        f"Welcome, **{st.session_state.username}**!"
    )

    st.write(
        "View your account information and your "
        "cotton seed prediction statistics."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Logged-in Username",
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
        item["Quality"] == "Good"
        for item in history
    )

    broken_count = sum(
        item["Quality"] == "Broken"
        for item in history
    )

    cracked_count = sum(
        item["Quality"] == "Cracked"
        for item in history
    )

    c1, c2 = st.columns(2)
    c1.metric("Total Predictions", total)
    c2.metric("Good Seeds", good_count)

    c3, c4 = st.columns(2)
    c3.metric("Broken Seeds", broken_count)
    c4.metric("Cracked Seeds", cracked_count)

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
            "No predictions yet. Open Seed Prediction "
            "to analyze a cotton seed image."
        )

# ==================================================
# 10. ADMIN DASHBOARD
# ==================================================
elif page == "Admin Dashboard":

    st.title("🛡️ Admin Dashboard")

    st.write(
        f"Welcome, Administrator "
        f"**{st.session_state.username}**!"
    )

    st.subheader("Current Session Statistics")

    history = st.session_state.history

    total = len(history)

    good_count = sum(
        item["Quality"] == "Good"
        for item in history
    )

    broken_count = sum(
        item["Quality"] == "Broken"
        for item in history
    )

    cracked_count = sum(
        item["Quality"] == "Cracked"
        for item in history
    )

    c1, c2 = st.columns(2)
    c1.metric("Total Predictions", total)
    c2.metric("Good Seeds", good_count)

    c3, c4 = st.columns(2)
    c3.metric("Broken Seeds", broken_count)
    c4.metric("Cracked Seeds", cracked_count)

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
        st.info(
            "No predictions recorded in this session."
        )

    st.caption(
        "Records are session-based, not stored in "
        "a permanent database."
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
        [
            "Capture Photo",
            "Upload Existing Image"
        ],
        horizontal=True
    )

    image_file = None

    # -------- CAPTURE PHOTO --------
    if input_method == "Capture Photo":

        image_file = st.camera_input(
            "Take a photo of the cotton seeds"
        )

    # -------- UPLOAD IMAGE --------
    else:

        image_file = st.file_uploader(
            "Choose a cotton seed image",
            type=["jpg", "jpeg", "png"]
        )

    # -------- PROCESS IMAGE --------
    if image_file is not None:

        try:
            image = Image.open(
                image_file
            ).convert("RGB")

            st.image(
                image,
                caption="Selected Cotton Seed Image",
                use_container_width=True
            )

            if st.button(
                "Predict Seed Quality",
                use_container_width=True
            ):

                with st.spinner(
                    "Analyzing cotton seed image..."
                ):

                    model = load_model()

                    resized_image = image.resize(
                        (224, 224)
                    )

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

                    # Must match the training class order
                    classes = [
                        "Broken",
                        "Cracked",
                        "Good"
                    ]

                    predicted_index = int(
                        np.argmax(predictions)
                    )

                    predicted_class = classes[
                        predicted_index
                    ]

                    confidence = float(
                        predictions[predicted_index] * 100
                    )

                # Save current user's prediction
                record = {
                    "Username": st.session_state.username,
                    "Role": st.session_state.role.title(),
                    "Quality": predicted_class,
                    "Confidence": f"{confidence:.2f}%"
                }

                st.session_state.history.append(record)

                st.success("Prediction completed!")

                st.subheader("Prediction Result")

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

                st.divider()
                st.subheader("Class Confidence Scores")

                for index, class_name in enumerate(classes):

                    score = float(
                        predictions[index] * 100
                    )

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

    # -------- RECENT PREDICTIONS --------
    st.divider()
    st.subheader("Recent Predictions")

    if st.session_state.history:

        st.dataframe(
            st.session_state.history,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "Your predictions will appear here after analysis."
        )
