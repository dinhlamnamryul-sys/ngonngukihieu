import streamlit as st
import os
from PIL import Image
from gtts import gTTS
import uuid
from unidecode import unidecode
import speech_recognition as sr

# ==============================
# CẤU HÌNH
# ==============================
DATA_DIR = "data"

HMONG_DICT = {
    "xin chào": "nyob zoo",
    "gia đình": "tsev neeg",
    "động vật": "tsiaj",
    "trái cây": "txiv hmab txiv ntoo",
    "a": "a",
    "đ": "đ",
    "0": "xoom",
    "1": "ib",
    "2": "ob"
}

# ==============================
# AI CORE (TỰ PHÂN BIỆT)
# ==============================
def ai_recognize(image):
    """
    AI giả lập – thay bằng model thật sau
    """
    return "A"   # ví dụ raw label

def ai_postprocess(label):
    """
    Tự phân biệt chữ / số / từ
    """
    for folder in os.listdir(DATA_DIR):
        if label in os.listdir(os.path.join(DATA_DIR, folder)):
            return folder, label
    return "unknown", label

# ==============================
# TÌM MEDIA
# ==============================
def find_media(label):
    label_norm = unidecode(label).lower()

    for folder in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder)
        for file in os.listdir(folder_path):
            name = os.path.splitext(file)[0]
            if unidecode(name).lower() == label_norm:
                return os.path.join(folder_path, file)
    return None

# ==============================
# TTS
# ==============================
def speak(text):
    file = f"tts_{uuid.uuid4().hex}.mp3"
    gTTS(text=text, lang="vi").save(file)
    st.audio(file)
    os.remove(file)

# ==============================
# TRANSLATE
# ==============================
def translate(text, lang):
    if lang == "Tiếng Mông":
        return HMONG_DICT.get(text.lower(), text)
    return text

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config("NNKH AI", layout="wide")
st.title("🤟 NGÔN NGỮ KÝ HIỆU AI – TỰ PHÂN BIỆT")

lang = st.selectbox("Ngôn ngữ xuất", ["Tiếng Việt", "Tiếng Mông"])

st.subheader("📷 Camera")

img_file = st.camera_input("Bật camera")

if img_file:
    img = Image.open(img_file)

    raw_label = ai_recognize(img)
    category, label = ai_postprocess(raw_label)

    label = translate(label, lang)

    st.success(f"AI nhận diện: {label} ({category})")

    media = find_media(label)
    if media:
        st.video(media)

    speak(label)

# ==============================
# VOICE SEARCH
# ==============================
st.subheader("🎙️ Tìm kiếm bằng giọng nói")

audio = st.audio_input("Nói")

if audio:
    r = sr.Recognizer()
    with sr.AudioFile(audio) as src:
        text = r.recognize_google(r.record(src), language="vi-VN")

    st.info(f"Bạn nói: {text}")
    media = find_media(text)
    if media:
        st.video(media)
