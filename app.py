
import streamlit as st
import tensorflow as tf
import numpy as np
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
if "history" not in st.session_state:
    st.session_state.history = []
st.title("🌱 Cotton Seed Quality Detection")
st.write("Upload a cotton seed image to predict its quality.")

uploaded_file = st.file_uploader(
    "Choose a cotton seed image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Selected Cotton Seed Image",
             use_container_width=True)

    if st.button("Predict Seed Quality"):
        with st.spinner("Analyzing image..."):
            model = load_model()

            image_resized = image.resize((224, 224))
            image_array = np.array(
                image_resized, dtype=np.float32
            )
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(image_array, verbose=0)[0]

            classes = ["Broken", "Cracked", "Good"]
            predicted_index = int(np.argmax(predictions))
            predicted_class = classes[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

        
st.subheader("Prediction Result")

if predicted_class == "Good":
    st.success("Quality: GOOD")
    st.write("The model classified this image as a good cotton seed.")
elif predicted_class == "Broken":
    st.warning("Quality: BROKEN")
    st.write("The model detected a broken cotton seed.")
else:
    st.warning("Quality: CRACKED")
    st.write("The model detected a cracked cotton seed.")

st.metric("Prediction Confidence", f"{confidence:.2f}%")
st.session_state.history.append({
    "Quality": predicted_class,
    "Confidence": f"{confidence:.2f}%"
})

st.subheader("Confidence by Class")

for i, class_name in enumerate(classes):
    score = float(predictions[i])
    st.write(f"**{class_name}:** {score * 100:.2f}%")
    st.progress(max(0.0, min(1.0, score)))

st.caption(
    "This is an AI prediction. Verify results against actual seed quality."
)

st.divider()
st.subheader("📋 Prediction History")

if st.session_state.history:
    st.dataframe(
        st.session_state.history,
        use_container_width=True
    )

    if st.button("Clear Prediction History"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("No predictions yet. Upload an image to get started.")
