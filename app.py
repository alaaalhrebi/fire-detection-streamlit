import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time
import threading
import gdown
import os

#  الصفحة
st.set_page_config(page_title="🔥 Fire Detection App", layout="centered")
st.title("🔥 Fire Detection AI System")

# تحميل النموذج من Google Drive 
model_filename = 'fire_detection_model.h5'
file_id = '1CEI7wUXISLEoAfXlE2HNl23TzcqHroLe'  # ← ضع هنا File ID الخاص بك

if not os.path.exists(model_filename):
    with st.spinner('📥 Downloading model...'):
        url = f'https://drive.google.com/uc?id=1CEI7wUXISLEoAfXlE2HNl23TzcqHroLe'
        gdown.download(url, model_filename, quiet=False)
    st.success('✅ Model downloaded successfully!')

# تحميل النموذج مع التخزين المؤقت
@st.cache_resource
def load_fire_model():
    return load_model(model_filename)

model = load_fire_model()

#  صوت الإنذار
#def play_alarm():
 #   playsound('alarm.mp3')

# زر لبدء الكاميرا
start_camera = st.button("🚨 Start Camera Detection")

if start_camera:
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            status_placeholder.error("❌ Unable to read from camera.")
            break

        img_resized = cv2.resize(frame, (224, 224))
        img_normalized = img_resized / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)

        prediction = model.predict(img_input)[0][0]

        label = "🔥 FIRE DETECTED!" if prediction > 0.5 else "✅ No Fire"
        color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)

        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        if prediction > 0.5 and not detected:
            detected = True
            status_placeholder.warning("🚨 FIRE DETECTED!")
            #threading.Thread(target=play_alarm).start()
            break

        time.sleep(1)

    cap.release()
    uploaded_photo = st.camera_input("📷 Take a picture")

option = st.radio("📸 Choose input method:", ["Upload Image", "Take Photo"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    image_source = uploaded_file

elif option == "Take Photo":
    uploaded_photo = st.camera_input("📷 Take a picture")
    image_source = uploaded_photo

else:
    image_source = None



if image_source is not None:
    image = Image.open(image_source).convert('RGB')
    st.image(image, caption='Input Image', use_column_width=True)

    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_input)[0][0]

    if prediction > 0.5:
        st.error("🚨 FIRE DETECTED!")
    else:
        st.success("✅ No Fire Detected.")
