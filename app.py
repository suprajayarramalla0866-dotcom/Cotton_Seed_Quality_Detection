
import streamlit as st
import tensorflow as tf
import numpy as np
import hmac
from PIL import Image

st.set_page_config(
    page_title="Cotton Seed Quality Detection",
    page_icon="🌱",
    layout="centered"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "cotton_seed_mobilenetv3.keras"
    )


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "role" not in st.session_state:
    st.session_state.role = ""

if "history" not in st.session_state:
    st.session_state.history = []


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



if not st.session_state.logged_in:
    st.markdown(
        """
        <div style="text-align:center; padding:20px;
                    background:linear-gradient(135deg,#e8f5e9,#ffffff);
                    border-radius:16px; border:1px solid #b7dfb9;">
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
        role = login_user(
            username.strip(), password
        )

        if role:
            st.session_state.logged_in = True
            st.session_state.username = username.strip()
            st.session_state.role = role
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.stop()



st.markdown(
    """
    <div style="padding:18px 20px;
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

if role == "admin":
    page = st.sidebar.radio(
        "Navigation",
        ["Admin Dashboard", "Seed Prediction"]
    )
else:
    page = "Seed Prediction"

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.rerun()


if page == "Admin Dashboard":
    st.subheader("🛡️ Administrator Dashboard")
    st.write("Login role: Administrator")

    st.metric(
        "Predictions in this session",
        len(st.session_state.history)
    )

    if st.session_state.history:
        st.subheader("Recent Prediction History")
        st.dataframe(
            st.session_state.history,
            use_container_width=True
        )

        if st.button("Clear Prediction History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions recorded in this session.")

else:
    st.subheader("Seed Quality Prediction")

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

        if st.button("Predict Seed Quality"):
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

            st.subheader("Prediction Result")

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
    else:
        st.info("No predictions yet. Upload an image to begin.")
