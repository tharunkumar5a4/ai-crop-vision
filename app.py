import streamlit as st
import numpy as np
import cv2
from tensorflow.keras import layers, models
import os
import gdown

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="AI Crop Vision", layout="wide")

# --------------------------------------------------
# DOWNLOAD MODEL FROM GOOGLE DRIVE (FIRST RUN ONLY)
# --------------------------------------------------
MODEL_PATH = "leaf_disease_model.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1pN7n2UlbgTXGt8PvlvKvAlhEKcyNGS_D"
    with st.spinner("🔄 Downloading AI model... please wait"):
        gdown.download(url, MODEL_PATH, quiet=False)

# --------------------------------------------------
# ANIMATED GLASS UI CSS
# --------------------------------------------------
st.markdown("""
<style>

/* Animated Gradient Background */
body {
    background: linear-gradient(-45deg, #1e3c72, #2a5298, #0f2027, #203a43);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Glass container */
.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    color: white;
}

/* Buttons */
.stButton>button {
    border-radius: 12px;
    background: linear-gradient(90deg,#ff512f,#dd2476);
    color: white;
    font-weight: bold;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATES
# --------------------------------------------------
if "user" not in st.session_state:
    st.session_state.user = None

if "language" not in st.session_state:
    st.session_state.language = "English"

# --------------------------------------------------
# LOGIN / SIGNUP
# --------------------------------------------------
def auth_page():
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.title("🌿 AI Crop Vision")

    tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == "admin" and password == "1234":
                st.session_state.user = username
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        st.text_input("Create Username")
        st.text_input("Create Password", type="password")
        if st.button("Create Account"):
            st.success("Account created (Demo Mode)")

    st.markdown('</div>', unsafe_allow_html=True)

if not st.session_state.user:
    auth_page()
    st.stop()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("🌿 Dashboard")
st.sidebar.write(f"Welcome, {st.session_state.user}")

if st.sidebar.button("🚪 Logout"):
    st.session_state.user = None
    st.rerun()

st.session_state.language = st.sidebar.selectbox(
    "🌐 Select Language",
    ["English", "Telugu"]
)

# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------
@st.cache_resource
def load_trained_model():
    model = models.Sequential([
        layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(64,activation='relu'),
        layers.Dense(3,activation='softmax')
    ])
    model.load_weights(MODEL_PATH)
    return model

model = load_trained_model()

classes = [
    "Potato Early Blight",
    "Potato Healthy",
    "Potato Late Blight"
]

# --------------------------------------------------
# REMEDIES WITH LANGUAGE SUPPORT
# --------------------------------------------------
remedies = {
    "Potato Early Blight": {
        "English": """
### 🔍 Disease Overview
Early Blight is a fungal disease caused by *Alternaria solani*.

### ⚠ Symptoms
- Brown circular spots on leaves  
- Yellow halo around lesions  
- Premature leaf drop  

### ✅ Solutions
- Remove infected leaves immediately  
- Apply copper-based fungicide weekly  
- Improve air circulation  
- Avoid overhead watering  
- Practice crop rotation  
""",
        "Telugu": """
### 🔍 వ్యాధి వివరాలు
ఎర్లీ బ్లైట్ ఒక ఫంగస్ వల్ల వచ్చే వ్యాధి.

### ⚠ లక్షణాలు
- ఆకులపై గోధుమ రంగు మచ్చలు  
- పసుపు రంగు చుట్టూ ఉండటం  
- ఆకులు ముందుగా రాలిపోవడం  

### ✅ పరిష్కారాలు
- బాధిత ఆకులను తొలగించండి  
- కాపర్ ఫంగిసైడ్ వాడండి  
- గాలి ప్రవాహం మెరుగుపరచండి  
- పై నుండి నీరు పోయవద్దు  
- పంట మార్పిడి పాటించండి  
"""
    },

    "Potato Healthy": {
        "English": """
### 🌿 Plant Status
The plant appears healthy.

### ✅ Maintenance Tips
- Maintain balanced fertilization  
- Monitor weekly  
- Ensure proper irrigation  
- Preventive fungicide if needed  
""",
        "Telugu": """
### 🌿 మొక్క స్థితి
మొక్క ఆరోగ్యంగా ఉంది.

### ✅ సంరక్షణ సూచనలు
- సమతుల్య ఎరువులు వాడండి  
- వారానికి పరిశీలించండి  
- సరైన నీరు ఇవ్వండి  
- అవసరమైతే ఫంగిసైడ్ వాడండి  
"""
    },

    "Potato Late Blight": {
        "English": """
### 🔍 Disease Overview
Late Blight is caused by *Phytophthora infestans*.

### ⚠ Symptoms
- Dark water-soaked lesions  
- Rapid spread in wet weather  
- Leaf wilting  

### ✅ Solutions
- Remove infected plants immediately  
- Apply systemic fungicide  
- Improve soil drainage  
- Avoid excess moisture  
- Maintain spacing between plants  
""",
        "Telugu": """
### 🔍 వ్యాధి వివరాలు
లేట్ బ్లైట్ ఒక తీవ్రమైన ఫంగస్ వ్యాధి.

### ⚠ లక్షణాలు
- నలుపు నీటి మచ్చలు  
- తేమ ఎక్కువగా ఉన్నప్పుడు వేగంగా వ్యాపిస్తుంది  
- ఆకులు వాడిపోవడం  

### ✅ పరిష్కారాలు
- బాధిత మొక్కలను తొలగించండి  
- సిస్టమిక్ ఫంగిసైడ్ వాడండి  
- నేలలో నీరు నిల్వ కాకుండా చూడండి  
- తేమ నియంత్రించండి  
- మొక్కల మధ్య దూరం ఉంచండి  
"""
    }
}

# --------------------------------------------------
# MAIN UI
# --------------------------------------------------
st.markdown('<div class="glass">', unsafe_allow_html=True)

title_text = "🥔 Smart Crop Disease Detection" if st.session_state.language == "English" else "🥔 స్మార్ట్ పంట వ్యాధి గుర్తింపు"
upload_text = "Upload Leaf Image" if st.session_state.language == "English" else "ఆకుల ఫోటో అప్‌లోడ్ చేయండి"
prediction_text = "Prediction" if st.session_state.language == "English" else "ఫలితం"
confidence_text = "Confidence" if st.session_state.language == "English" else "నమ్మక స్థాయి"
remedy_text = "Recommended Remedy" if st.session_state.language == "English" else "సూచించిన పరిష్కారం"

st.title(title_text)

uploaded_file = st.file_uploader(upload_text, type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, use_container_width=True)

    img_resized = cv2.resize(img,(128,128))/255.0
    img_resized = np.expand_dims(img_resized,0)

    prediction = model.predict(img_resized)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction)*100

    st.markdown(f"### 🧠 {prediction_text}")
    st.success(predicted_class)

    st.markdown(f"### 📊 {confidence_text}")
    st.progress(int(confidence))
    st.write(f"{confidence:.2f}%")

    st.markdown(f"### 🌱 {remedy_text}")
    st.markdown(remedies[predicted_class][st.session_state.language])

st.markdown('</div>', unsafe_allow_html=True)