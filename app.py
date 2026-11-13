import streamlit as st
import sqlite3
import hashlib
import secrets
import hmac
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image

# =========================================================
# 1. APP CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="CottonIQ | Seed Quality Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "cotton_seed_mobilenetv3.keras"
DB_PATH = BASE_DIR / "users.db"

CLASS_NAMES = ["Broken", "Cracked", "Good"]
IMG_SIZE = (224, 224)

# =========================================================
# 2. PROFESSIONAL GREEN THEME
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --green-950: #123524;
    --green-900: #17452f;
    --green-800: #1d5b3b;
    --green-700: #26734a;
    --green-100: #e9f4ec;
    --green-50: #f5faf6;
    --ink: #1d2b22;
    --muted: #718078;
    --line: #e5ece7;
    --white: #ffffff;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f5f8f5;
    color: var(--ink);
}

.block-container {
    max-width: 1450px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #edf5ef;
    border-right: 1px solid #dfe9e1;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

.brand {
    display:flex;
    align-items:center;
    gap:12px;
    padding: 10px 4px 22px 4px;
}

.brand-mark {
    width:46px;
    height:46px;
    border-radius:14px;
    display:flex;
    align-items:center;
    justify-content:center;
    background:#d8ebdd;
    font-size:25px;
}

.brand-name {
    font-family:'Manrope',sans-serif;
    font-size:20px;
    font-weight:800;
    color:#17452f;
    line-height:1.1;
}

.brand-sub {
    font-size:11px;
    letter-spacing:1.5px;
    text-transform:uppercase;
    color:#718078;
    margin-top:4px;
}

.page-eyebrow {
    color:#26734a;
    font-weight:800;
    font-size:11px;
    letter-spacing:2px;
    text-transform:uppercase;
    margin-bottom:7px;
}

.page-title {
    font-family:'Manrope',sans-serif;
    color:#173d2b;
    font-size:32px;
    font-weight:800;
    line-height:1.2;
    margin:0;
}

.page-subtitle {
    color:#718078;
    font-size:14px;
    margin-top:7px;
    margin-bottom:22px;
}

.hero {
    background: linear-gradient(
        120deg,
        #17452f 0%,
        #26734a 65%,
        #3d895a 100%
    );
    padding:30px 32px;
    border-radius:22px;
    color:white;
    box-shadow:0 12px 30px rgba(23,69,47,.12);
}

.hero h1 {
    color:white;
    font-family:'Manrope',sans-serif;
    font-size:30px;
    margin:0 0 8px 0;
}

.hero p {
    color:#e2f0e5;
    font-size:14px;
    margin:0;
    max-width:650px;
}

.hero-tag {
    display:inline-block;
    background:rgba(255,255,255,.14);
    border:1px solid rgba(255,255,255,.22);
    border-radius:30px;
    padding:6px 12px;
    font-size:11px;
    letter-spacing:1px;
    margin-bottom:15px;
    color:#f4fff6;
}

.section-title {
    color:#173d2b;
    font-family:'Manrope',sans-serif;
    font-weight:800;
    font-size:19px;
    margin:8px 0 12px 0;
}

.info-card {
    background:white;
    border:1px solid #e5ece7;
    border-radius:17px;
    padding:18px 19px;
    min-height:118px;
    box-shadow:0 4px 15px rgba(25,55,35,.035);
}

.info-label {
    color:#718078;
    font-size:12px;
    font-weight:600;
    margin-bottom:9px;
}

.info-value {
    color:#173d2b;
    font-family:'Manrope',sans-serif;
    font-size:27px;
    font-weight:800;
}

.info-note {
    color:#8b9890;
    font-size:11px;
    margin-top:5px;
}

.panel {
    background:white;
    border:1px solid #e5ece7;
    border-radius:18px;
    padding:20px 22px;
    box-shadow:0 4px 15px rgba(25,55,35,.035);
}

.result-good {
    background:#e9f6ed;
    border:1px solid #cce8d4;
    border-radius:16px;
    padding:18px;
}

.result-warn {
    background:#fff5e8;
    border:1px solid #f2dfc1;
    border-radius:16px;
    padding:18px;
}

.result-bad {
    background:#fff0ed;
    border:1px solid #f0d4ce;
    border-radius:16px;
    padding:18px;
}

.result-name {
    font-family:'Manrope',sans-serif;
    font-size:25px;
    font-weight:800;
    margin:3px 0;
}

.small-muted {
    color:#718078;
    font-size:12px;
}

.stButton > button {
    background:#1d5b3b;
    color:white;
    border:0;
    border-radius:10px;
    min-height:44px;
    font-weight:700;
    transition:all .2s ease;
}

.stButton > button:hover {
    background:#26734a;
    color:white;
    border:0;
    transform:translateY(-1px);
}

.stButton > button:focus {
    box-shadow:0 0 0 2px #b8d9c1;
}

/* =====================================================
   HIGHLIGHTED CSV DOWNLOAD BUTTON
   ===================================================== */

[data-testid="stDownloadButton"] button {
    background: #f4c542 !important;
    color: #173d2b !important;
    border: 2px solid #d69e00 !important;
    border-radius: 12px !important;
    min-height: 48px !important;
    font-weight: 800 !important;
    box-shadow: 0 5px 14px rgba(214, 158, 0, 0.22) !important;
    transition: all 0.2s ease !important;
}

[data-testid="stDownloadButton"] button:hover {
    background: #ffda63 !important;
    color: #123524 !important;
    border-color: #bd8500 !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(214, 158, 0, 0.30) !important;
}

[data-testid="stDownloadButton"] button:focus {
    outline: 3px solid rgba(214, 158, 0, 0.35) !important;
    outline-offset: 2px !important;
}

.stTextInput input,
.stSelectbox div[data-baseweb="select"] {
    border-radius:10px;
}

[data-testid="stMetric"] {
    background:white;
    border:1px solid #e5ece7;
    border-radius:15px;
    padding:15px 17px;
    box-shadow:0 4px 15px rgba(25,55,35,.035);
}

[data-testid="stMetricLabel"] {
    color:#718078;
}

[data-testid="stMetricValue"] {
    color:#173d2b;
    font-family:'Manrope',sans-serif;
}

div[data-testid="stFileUploader"] {
    background:#fbfdfb;
    border:1px dashed #a9c9b1;
    border-radius:14px;
    padding:10px;
}

.stProgress > div > div > div > div {
    background:#26734a;
}

hr {
    border-color:#e5ece7;
}

footer {
    visibility:hidden;
}

#MainMenu {
    visibility:hidden;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. DATABASE AND ACCOUNT FUNCTIONS
# =========================================================

def get_connection():
    return sqlite3.connect(DB_PATH)


def initialize_database():
    with get_connection() as conn:
        # Keep the existing users table and account records unchanged.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL
            )
        """)

        # Store predictions permanently instead of keeping them only
        # in Streamlit session_state.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                prediction_time TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                image_name TEXT
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_username
            ON predictions(username)
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
            conn.execute("""
                INSERT INTO users (
                    username,
                    password_hash,
                    salt,
                    role
                )
                VALUES (?, ?, ?, ?)
            """, (username, password_hash, salt, role))

        return True, "Account created successfully. You can now log in."

    except sqlite3.IntegrityError:
        return False, "Username already exists. Choose another."


def check_login(username, password, selected_role):
    with get_connection() as conn:
        result = conn.execute("""
            SELECT password_hash, salt, role
            FROM users
            WHERE username = ?
        """, (username.strip(),)).fetchone()

    if result is None:
        return False

    stored_hash, salt, stored_role = result
    entered_hash = hash_password(password, salt)

    return (
        hmac.compare_digest(stored_hash, entered_hash)
        and stored_role.lower() == selected_role.lower()
    )


def initialize_admin():
    """Create admin account from Streamlit Secrets, if configured."""

    try:
        admin_username = st.secrets["ADMIN_USERNAME"]
        admin_password = st.secrets["ADMIN_PASSWORD"]

    except Exception:
        return

    if not admin_username or not admin_password:
        return

    with get_connection() as conn:
        exists = conn.execute(
            "SELECT username FROM users WHERE username = ?",
            (admin_username.strip(),)
        ).fetchone()

    if exists is None:
        create_user(admin_username, admin_password, "admin")


# =========================================================
# 4. SESSION STATE
# =========================================================

def initialize_session():
    defaults = {
        "logged_in": False,
        "username": "",
        "role": "",
        "show_login": False,
        "history": [],
        "last_result": None,
        "navigation": "Overview"
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =========================================================
# 5. REUSABLE UI HELPERS
# =========================================================

def brand_block():
    st.markdown("""
    <div class="brand">
      <div class="brand-mark">🌱</div>
      <div>
        <div class="brand-name">CottonIQ</div>
        <div class="brand-sub">Seed intelligence</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def page_heading(eyebrow, title, subtitle):
    st.markdown(
        f'<div class="page-eyebrow">{eyebrow}</div>'
        f'<div class="page-title">{title}</div>'
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )


def metric_card(label, value, note, icon):
    st.markdown(f"""
    <div class="info-card">
      <div class="info-label">{icon} &nbsp; {label}</div>
      <div class="info-value">{value}</div>
      <div class="info-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def show_top_header():
    st.markdown("""
    <div style="display:flex;justify-content:space-between;
                align-items:center;gap:12px;margin-bottom:18px;">
      <div>
        <div style="font-size:11px;letter-spacing:1.7px;
                    color:#26734a;font-weight:800;">
          SMART SEEDS. BETTER HARVEST.
        </div>
        <div style="font-size:13px;color:#718078;margin-top:3px;">
          AI-powered cotton seed quality inspection
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 6. WELCOME AND AUTHENTICATION
# =========================================================

def welcome_page():
    st.markdown(
        "<div style='height:5vh'></div>",
        unsafe_allow_html=True
    )

    left, center, right = st.columns([1, 2.4, 1])

    with center:
        st.markdown("""
        <div class="hero">
          <div class="hero-tag">AI-POWERED QUALITY INSPECTION</div>
          <div style="font-size:48px;margin-bottom:10px;">🌱</div>
          <h1>Smart Seeds.<br>Better Harvest.</h1>
          <p>
            Inspect cotton seed quality with image-based classification.
            Get clear results for Good, Broken and Cracked seed categories.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        a, b, c = st.columns([1, 2, 1])

        with b:
            if st.button("Get Started  →", use_container_width=True):
                st.session_state.show_login = True
                st.rerun()

        st.markdown(
            "<p style='text-align:center;color:#8b9890;font-size:12px;'>"
            "COTTONIQ · SEED QUALITY INTELLIGENCE</p>",
            unsafe_allow_html=True
        )


def login_page():
    brand_block()
    show_top_header()

    left, center, right = st.columns([1, 1.15, 1])

    with center:
        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True
        )

        st.markdown("### Welcome back")
        st.caption(
            "Sign in to continue to your inspection workspace."
        )

        login_tab, signup_tab = st.tabs(
            ["Sign in", "Create account"]
        )

        with login_tab:
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
                "Account type",
                ["User", "Admin"],
                horizontal=True,
                key="login_role"
            )

            if st.button("Sign in", use_container_width=True):
                if check_login(username, password, role):
                    st.session_state.logged_in = True
                    st.session_state.username = username.strip()
                    st.session_state.role = role.lower()

                    st.session_state.navigation = (
                        "Admin Overview"
                        if role.lower() == "admin"
                        else "Overview"
                    )

                    st.rerun()

                else:
                    st.error(
                        "Invalid username, password, or account type."
                    )

        with signup_tab:
            new_username = st.text_input(
                "Choose a username",
                key="signup_username"
            )

            new_password = st.text_input(
                "Create a password",
                type="password",
                key="signup_password"
            )

            confirm_password = st.text_input(
                "Confirm password",
                type="password",
                key="signup_confirm"
            )

            if st.button(
                "Create account",
                use_container_width=True
            ):
                if len(new_password) < 8:
                    st.error(
                        "Password must contain at least 8 characters."
                    )

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

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# =========================================================
# 7. MODEL LOADING AND PREDICTION
# =========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Place it in the same folder as app.py."
        )

    import tensorflow as tf

    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image):
    image = image.convert("RGB").resize(IMG_SIZE)

    array = np.asarray(
        image,
        dtype=np.float32
    )

    return np.expand_dims(array, axis=0)


def predict_seed(image):
    model = load_model()

    processed = preprocess_image(image)

    predictions = model.predict(
        processed,
        verbose=0
    )

    scores = np.asarray(
        predictions[0],
        dtype=float
    )

    if len(scores) != len(CLASS_NAMES):
        raise ValueError(
            "Model output count does not match CLASS_NAMES. "
            "Verify the class order used during training."
        )

    predicted_class = CLASS_NAMES[int(np.argmax(scores))]

    return predicted_class, scores


# =========================================================
# 8. HISTORY HELPERS
# =========================================================

def current_user_history():
    """Read all permanently saved predictions for the logged-in user."""
    username = st.session_state.username

    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT username, prediction_time, prediction,
                   confidence, image_name
            FROM predictions
            WHERE username = ?
            ORDER BY id ASC
        """, (username,)).fetchall()

    return [
        {
            "username": row["username"],
            "time": row["prediction_time"],
            "prediction": row["prediction"],
            "confidence": float(row["confidence"]),
            "image": row["image_name"] or "—"
        }
        for row in rows
    ]


def all_history():
    """Read permanently saved predictions for every account."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT username, prediction_time, prediction,
                   confidence, image_name
            FROM predictions
            ORDER BY id ASC
        """).fetchall()

    return [
        {
            "username": row["username"],
            "time": row["prediction_time"],
            "prediction": row["prediction"],
            "confidence": float(row["confidence"]),
            "image": row["image_name"] or "—"
        }
        for row in rows
    ]


def save_history(predicted_class, scores, image_name):
    """Save a prediction immediately to the permanent SQLite database."""
    username = st.session_state.get("username", "")

    if not username:
        raise ValueError("No logged-in username is available to save this prediction.")

    confidence = float(np.max(scores)) * 100
    prediction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO predictions (
                username, prediction_time, prediction,
                confidence, image_name
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            username,
            prediction_time,
            predicted_class,
            round(confidence, 2),
            str(image_name or "—")
        ))


# =========================================================
# 9. USER OVERVIEW
# =========================================================

def user_overview():
    history = current_user_history()

    total = len(history)

    good = sum(
        x["prediction"] == "Good"
        for x in history
    )

    broken = sum(
        x["prediction"] == "Broken"
        for x in history
    )

    cracked = sum(
        x["prediction"] == "Cracked"
        for x in history
    )

    page_heading(
        "WORKSPACE / OVERVIEW",
        f"Welcome, {st.session_state.username}",
        "Here is a snapshot of your cotton seed inspection activity."
    )

    st.markdown("""
    <div class="hero">
      <div class="hero-tag">INSPECTION WORKSPACE</div>
      <h1>Make every seed check count.</h1>
      <p>
        Upload or capture a cotton seed image, run the AI classifier,
        and review the predicted quality category with its confidence score.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    cols = st.columns(4)

    metrics = [
        (
            "Total inspections",
            total,
            "Your recorded predictions",
            "◉"
        ),
        (
            "Good seeds",
            good,
            "Classified as Good",
            "✓"
        ),
        (
            "Broken seeds",
            broken,
            "Classified as Broken",
            "◇"
        ),
        (
            "Cracked seeds",
            cracked,
            "Classified as Cracked",
            "⌁"
        )
    ]

    for col, (label, value, note, icon) in zip(cols, metrics):
        with col:
            metric_card(
                label,
                value,
                note,
                icon
            )

    st.write("")

    left, right = st.columns([1.25, 1])

    with left:
        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">Quality distribution</div>',
            unsafe_allow_html=True
        )

        if total:
            st.bar_chart(
                {"Inspections": [good, broken, cracked]},
                horizontal=True
            )

            st.caption(
                "Counts from your saved prediction history."
            )

        else:
            st.info(
                "Your quality distribution will appear after your first inspection."
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with right:
        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">Quick start</div>',
            unsafe_allow_html=True
        )

        st.write("**01 ·** Open Seed Inspection.")
        st.write("**02 ·** Capture or upload a clear seed image.")
        st.write("**03 ·** Select Analyse Seed to view the result.")
        st.write("**04 ·** Review your confidence scores and history.")

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# =========================================================
# 10. SEED INSPECTION
# =========================================================

def seed_inspection():
    page_heading(
        "AI ANALYSIS / INSPECTION",
        "Seed inspection",
        "Capture a cotton seed image or upload an image from your device."
    )

    input_col, preview_col = st.columns(
        [1, 1],
        gap="large"
    )

    with input_col:
        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">1. Select an image</div>',
            unsafe_allow_html=True
        )

        source = st.radio(
            "Image source",
            ["Upload image", "Use camera"],
            horizontal=True,
            key="image_source"
        )

        image_file = None

        if source == "Upload image":
            image_file = st.file_uploader(
                "Choose a cotton seed image",
                type=["jpg", "jpeg", "png"],
                key="inspection_upload"
            )

        else:
            image_file = st.camera_input(
                "Capture cotton seed image",
                key="inspection_camera"
            )

        st.caption(
            "Tip: Use a focused image with good lighting and a visible seed."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with preview_col:
        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">2. Image preview</div>',
            unsafe_allow_html=True
        )

        if image_file is not None:
            try:
                image = Image.open(image_file).convert("RGB")

                st.image(
                    image,
                    use_container_width=True
                )

            except Exception:
                st.error(
                    "The selected file could not be opened as an image."
                )

                image = None

        else:
            image = None

            st.markdown("""
            <div style="height:245px;border:1px dashed #b7cdbd;
                        border-radius:14px;display:flex;
                        align-items:center;justify-content:center;
                        flex-direction:column;background:#fbfdfb;
                        color:#718078;">
              <div style="font-size:42px;">🖼️</div>
              <div style="font-weight:700;color:#365b43;margin-top:8px;">
                No image selected
              </div>
              <div style="font-size:12px;margin-top:4px;">
                Your image preview will appear here
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    if image_file is not None and image is not None:
        st.write("")

        if st.button(
            "✦  Analyse seed quality",
            use_container_width=True
        ):

            try:
                with st.spinner(
                    "Analysing image with MobileNetV3..."
                ):
                    predicted_class, scores = predict_seed(image)

                save_history(
                    predicted_class,
                    scores,
                    getattr(image_file, "name", "camera_capture")
                )

                st.session_state.last_result = {
                    "prediction": predicted_class,
                    "scores": [
                        float(x) for x in scores
                    ],
                    "confidence": float(np.max(scores)) * 100
                }

                st.rerun()

            except Exception as error:
                st.error(
                    "Prediction could not be completed."
                )

                st.exception(error)

    result = st.session_state.last_result

    if result is not None:
        st.write("")

        st.markdown(
            '<div class="section-title">3. Inspection result</div>',
            unsafe_allow_html=True
        )

        predicted = result["prediction"]
        confidence = result["confidence"]

        style_class = (
            "result-good" if predicted == "Good"
            else "result-warn" if predicted == "Cracked"
            else "result-bad"
        )

        symbol = (
            "✓" if predicted == "Good"
            else "!" if predicted == "Cracked"
            else "×"
        )

        st.markdown(f"""
        <div class="{style_class}">
          <div class="small-muted">AI PREDICTED CATEGORY</div>
          <div class="result-name">{symbol} &nbsp; {predicted}</div>
          <div style="font-size:14px;color:#52645a;">
            Model confidence: <b>{confidence:.2f}%</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        score_cols = st.columns(3)

        for col, class_name, score in zip(
            score_cols,
            CLASS_NAMES,
            result["scores"]
        ):

            with col:
                st.metric(
                    class_name,
                    f"{score * 100:.2f}%"
                )

                st.progress(
                    float(np.clip(score, 0, 1))
                )

        st.caption(
            "The displayed confidence is the model's class score, not a guarantee "
            "of seed quality. Confirm results with appropriate physical inspection."
        )


# =========================================================
# 11. PREDICTION HISTORY
# =========================================================

def prediction_history():
    page_heading(
        "RECORDS / HISTORY",
        "Inspection history",
        "Review predictions saved permanently in the database."
    )

    history = current_user_history()

    if not history:
        st.info(
            "No inspections recorded yet. Run your first seed inspection."
        )
        return

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Your recent inspections</div>',
        unsafe_allow_html=True
    )

    display_rows = [
        {
            "Date & time": x["time"],
            "Image": x.get("image", "—"),
            "Predicted class": x["prediction"],
            "Confidence": f'{x["confidence"]:.2f}%'
        }
        for x in reversed(history)
    ]

    st.dataframe(
        display_rows,
        use_container_width=True,
        hide_index=True
    )

    csv_data = (
        __import__("pandas")
        .DataFrame(display_rows)
        .to_csv(index=False)
        .encode("utf-8")
    )

    # Highlighted CSV download button
    st.download_button(
        label="⬇  Download history as CSV",
        data=csv_data,
        file_name="cotton_seed_inspection_history.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# 12. ADMIN OVERVIEW
# =========================================================

def admin_overview():
    page_heading(
        "ADMINISTRATION / OVERVIEW",
        "Admin control centre",
        "Monitor registered accounts and saved inspection activity."
    )

    with get_connection() as conn:
        users = conn.execute("""
            SELECT username, role
            FROM users
            ORDER BY username
        """).fetchall()

    history = all_history()

    total = len(history)

    good = sum(
        x["prediction"] == "Good"
        for x in history
    )

    broken = sum(
        x["prediction"] == "Broken"
        for x in history
    )

    cracked = sum(
        x["prediction"] == "Cracked"
        for x in history
    )

    cols = st.columns(4)

    metrics = [
        (
            "Registered accounts",
            len(users),
            "Accounts in users database",
            "♙"
        ),
        (
            "Total inspections",
            total,
            "All saved predictions",
            "◉"
        ),
        (
            "Good predictions",
            good,
            "Classified as Good",
            "✓"
        ),
        (
            "Damaged predictions",
            broken + cracked,
            "Broken + Cracked",
            "◇"
        )
    ]

    for col, (label, value, note, icon) in zip(cols, metrics):
        with col:
            metric_card(
                label,
                value,
                note,
                icon
            )

    st.write("")

    users_col, activity_col = st.columns(
        [1, 1.2]
    )

    with users_col:
        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">Registered accounts</div>',
            unsafe_allow_html=True
        )

        if users:
            st.dataframe(
                [
                    {
                        "Username": u,
                        "Role": r.title()
                    }
                    for u, r in users
                ],
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info(
                "No registered accounts found."
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with activity_col:
        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">Inspection activity</div>',
            unsafe_allow_html=True
        )

        if history:
            st.bar_chart(
                {
                    "Predictions": [
                        good,
                        broken,
                        cracked
                    ]
                },
                horizontal=True
            )

        else:
            st.info(
                "Inspection statistics will appear after predictions are made."
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    st.write("")

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Recent system predictions</div>',
        unsafe_allow_html=True
    )

    if history:
        st.dataframe(
            [
                {
                    "Username": x["username"],
                    "Date & time": x["time"],
                    "Prediction": x["prediction"],
                    "Confidence": f'{x["confidence"]:.2f}%'
                }
                for x in reversed(history)
            ],
            use_container_width=True,
            hide_index=True
        )

    else:
        st.caption(
            "No saved predictions are available."
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# 13. NAVIGATION AND LOGOUT
# =========================================================

def sidebar_navigation():
    with st.sidebar:
        brand_block()

        st.markdown(
            f"<div style='font-size:13px;color:#52645a;'>"
            f"Signed in as <b>{st.session_state.username}</b></div>",
            unsafe_allow_html=True
        )

        st.caption(
            f"Account: {st.session_state.role.title()}"
        )

        st.divider()

        if st.session_state.role == "admin":
            pages = ["Admin Overview"]

        else:
            pages = [
                "Overview",
                "Seed Inspection",
                "Inspection History"
            ]

        current = st.session_state.navigation

        if current not in pages:
            current = pages[0]

        selected = st.radio(
            "WORKSPACE",
            pages,
            index=pages.index(current),
            key="main_navigation_radio"
        )

        st.session_state.navigation = selected

        st.divider()

        st.markdown(
            "<div style='font-size:11px;color:#718078;line-height:1.6;'>"
            "COTTONIQ<br>Smart Seeds. Better Harvest.</div>",
            unsafe_allow_html=True
        )

        if st.button(
            "↪  Sign out",
            use_container_width=True
        ):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.role = ""
            st.session_state.show_login = False
            st.session_state.last_result = None
            st.session_state.navigation = "Overview"

            st.rerun()

        return selected


# =========================================================
# 14. MAIN APPLICATION
# =========================================================

def main():
    initialize_database()
    initialize_admin()
    initialize_session()

    if not st.session_state.logged_in:

        if st.session_state.show_login:
            login_page()

        else:
            welcome_page()

        return

    selected_page = sidebar_navigation()

    show_top_header()

    if st.session_state.role == "admin":
        admin_overview()

    elif selected_page == "Overview":
        user_overview()

    elif selected_page == "Seed Inspection":
        seed_inspection()

    elif selected_page == "Inspection History":
        prediction_history()


# =========================================================
# 15. APPLICATION ENTRY POINT
# =========================================================

if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        st.error(
            "The application encountered an unexpected error."
        )
        st.exception(error)
