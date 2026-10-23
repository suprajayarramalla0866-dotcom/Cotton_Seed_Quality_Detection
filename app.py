
import streamlit as st
import tensorflow as tf
import numpy as np
import hmac
from PIL import Image

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Cotton Seed Quality Detection",
    page_icon="🌱",
    layout="centered"
)

# --------------------------------------------------
# AGRICULTURE THEME STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
    }

    h1, h2, h3, h4 {
        color: #216e39;
    }

    div.stButton > button {
        background-color: #218838;
        color: white;
        border: 1px solid #218838;
        border-radius: 8px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #17652a;
        color: white;
        border: 1px solid #17652a;
    }

    div[data-testid="stSidebar"] {
        background-color: #edf7ed;
    }

    div[data-testid="stSidebar"] h1,
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3 {
        color: #17652a;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD AI MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "cotton_seed_mobilenetv3.keras"
    )

# --------------------------------------------------
# INITIALIZE SESSION STATE
# --------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "role" not in st.session_state:
    st.session_state.role = ""

if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------
# LOGIN FUNCTION
# --------------------------------------------------
def login_user(username, password):
    users = st.secrets["users"]

    for role_name, details in users.items():
        saved_username = str(details["username"])
        saved_password = str(details["password"])

        if (
            hmac.compare_digest(username, saved_username)
            and hmac.compare_digest(password, saved_password)
        ):
            return role_name

    return None

# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------
if not st.session_state.logged_in:
    st.markdown(
        """
        <div style="text-align:center; padding:24px;
                    background:linear-gradient(135deg,#e8f5e9,#ffffff);
                    border-radius:16px;
                    border:1px solid #b7dfb9;">
            <div style="font-size:48px;">🌿</div>
            <h1 style="color:#17652a; margin-bottom:4px;">
                Cotton Seed Quality Detection
            </h1>
            <p style="color:#367747; font-style:italic;">
                Smart Seeds. Better Harvest.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Login to continue")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input(
            "Password", type="password"
        )

        submitted = st.form_submit_button(
            "Login", use_container_width=True
        )

    if submitted:
        role = login_user(username.strip(), password)

        if role:
            st.session_state.logged_in = True
            st.session_state.username = username.strip()
            st.session_state.role = role
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.stop()

# --------------------------------------------------
# AGRICULTURE HEADER
# --------------------------------------------------
st.markdown(
    """
    <div style="padding:20px;
                background:linear-gradient(120deg,#14532d,#2e7d32);
                border-radius:14px;
                margin-bottom:20px;">
        <h1 style="color:white; margin:0;">
            🌿 Cotton Seed Quality Detection
        </h1>
        <p style="color:#e8f5e9; margin:8px 0 0 0;">
            Smart Seeds. Better Harvest.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

role = st.session_state.role

st.write(
    f"Welcome, **{st.session_state.username}**!"
)

# --------------------------------------------------
# ROLE-BASED NAVIGATION
# --------------------------------------------------
if role == "admin":
    page = st.sidebar.radio(
        "🌱 Navigation",
        [
            "Admin Dashboard",
            "User Dashboard",
            "Seed Prediction"
        ]
    )
else:
    page = st.sidebar.radio(
        "🌱 Navigation",
        [
            "User Dashboard",
            "Seed Prediction"
        ]
    )

# --------------------------------------------------
# GREEN LOGOUT BUTTON FOR BOTH ROLES
# --------------------------------------------------
st.sidebar.markdown(
    """
    <div style="padding:8px 0 4px 0;
                color:#216e39; font-weight:600;">
        Account
    </div>
    """,
    unsafe_allow_html=True
)

if st.sidebar.button(
    "🚪 Logout",
    key="logout_button",
    use_container_width=True
):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.session_state.history = []
    st.rerun()

# --------------------------------------------------
# USER DASHBOARD
# --------------------------------------------------
if page == "User Dashboard":
    st.markdown(
        '<h2 style="color:#216e39;">'
        '🌱 Seed Shop Owner Dashboard</h2>',
        unsafe_allow_html=True
    )

    st.write(
        "Welcome to your cotton seed quality workspace."
    )

    history = st.session_state.history

    total_predictions = len(history)
    good_count = sum(
        1 for item in history
        if item["Quality"] == "Good"
    )
    broken_count = sum(
        1 for item in history
        if item["Quality"] == "Broken"
    )
    cracked_count = sum(
        1 for item in history
        if item["Quality"] == "Cracked"
    )

    col1, col2 = st.columns(2)
    col1.metric("Total Predictions", total_predictions)
    col2.metric("Good Seeds", good_count)

    col3, col4 = st.columns(2)
    col3.metric("Broken Seeds", broken_count)
    col4.metric("Cracked Seeds", cracked_count)

    st.markdown(
        """
        <div style="padding:16px;
                    background:#edf7ed;
                    border:1px solid #c8e6c9;
                    border-radius:12px;">
            <h4 style="color:#216e39;">
                🌾 Start Seed Analysis
            </h4>
            <p style="color:#315b3a;">
                Select Seed Prediction from the sidebar to upload
                a cotton seed image and analyze its quality.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("📋 Prediction History")

    if st.session_state.history:
        st.dataframe(
            st.session_state.history,
            use_container_width=True
        )

        if st.button(
            "🗑️ Clear Prediction History",
            key="clear_user_history"
        ):
            st.session_state.history = []
            st.rerun()
    else:
        st.info(
            "No predictions yet. Start by analyzing a seed image."
        )

# --------------------------------------------------
# ADMIN DASHBOARD
# --------------------------------------------------
elif page == "Admin Dashboard":
    st.markdown(
        '<h2 style="color:#216e39;">'
        '🛡️ Administrator Dashboard</h2>',
        unsafe_allow_html=True
    )

    st.write("Login role: Administrator")

    st.metric(
        "Predictions in This Session",
        len(st.session_state.history)
    )

    if st.session_state.history:
        st.subheader("Recent Prediction History")

        st.dataframe(
            st.session_state.history,
            use_container_width=True
        )

        if st.button(
            "🗑️ Clear Prediction History",
            key="clear_admin_history"
        ):
            st.session_state.history = []
            st.rerun()
    else:
        st.info(
            "No predictions recorded in this session."
        )

# --------------------------------------------------
# SEED PREDICTION PAGE
# --------------------------------------------------
elif page == "Seed Prediction":
    st.markdown(
        """
        <h2 style="color:#216e39;">
            🌱 Seed Quality Prediction
        </h2>
        <p style="color:#50765a;">
            Upload a cotton seed image to analyze its quality.
        </p>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a cotton seed image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Selected Cotton Seed Image",
            use_container_width=True
        )

        if st.button(
            "🌿 Predict Seed Quality",
            key="predict_seed"
        ):
            with st.spinner("Analyzing image..."):
                model = load_model()

                resized = image.resize((224, 224))
                image_array = np.array(
                    resized, dtype=np.float32
                )
                image_array = np.expand_dims(
                    image_array, axis=0
                )

                predictions = model.predict(
                    image_array, verbose=0
                )[0]

                classes = ["Broken", "Cracked", "Good"]
                index = int(np.argmax(predictions))
                predicted_class = classes[index]
                confidence = float(
                    predictions[index]
                ) * 100

                st.session_state.history.append({
                    "Quality": predicted_class,
                    "Confidence": f"{confidence:.2f}%"
                })

            st.markdown(
                '<h2 style="color:#216e39;">'
                '🌿 Prediction Result</h2>',
                unsafe_allow_html=True
            )

            if predicted_class == "Good":
                st.success("Quality: GOOD")
            elif predicted_class == "Broken":
                st.warning("Quality: BROKEN")
            else:
                st.warning("Quality: CRACKED")

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.subheader("Confidence by Class")

            for i, name in enumerate(classes):
                score = float(predictions[i])

                st.write(
                    f"**{name}:** {score * 100:.2f}%"
                )

                st.progress(
                    max(0.0, min(1.0, score))
                )

    st.divider()
    st.subheader("📋 Prediction History")

    if st.session_state.history:
        st.dataframe(
            st.session_state.history,
            use_container_width=True
        )

        if st.button(
            "🗑️ Clear Prediction History",
            key="clear_prediction_history"
        ):
            st.session_state.history = []
            st.rerun()
    else:
        st.info(
            "No predictions yet. Upload an image to begin."
        )
