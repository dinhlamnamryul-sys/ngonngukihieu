import streamlit as st
import cv2
import mediapipe as mp
import os
import tempfile
import time
from unidecode import unidecode
from gtts import gTTS
from PIL import Image
import numpy as np

# --- Cấu hình MediaPipe ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_utils = mp.solutions.drawing_utils

# Style vẽ
pose_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
hand_style = drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)

# --- Hàm Tiện Ích ---
def process_frame(frame):
    """Xử lý nhận diện tư thế và tay trên một khung hình"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Xử lý nhận diện
    results_pose = pose.process(frame_rgb)
    results_hands = hands.process(frame_rgb)
    
    # Vẽ kết quả
    if results_pose.pose_landmarks:
        drawing_utils.draw_landmarks(frame_rgb, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS, pose_style, pose_style)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            drawing_utils.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS, hand_style, hand_style)
            
    return frame_rgb

def play_audio_st(text):
    """Tạo và phát âm thanh trong Streamlit"""
    tts = gTTS(text=f"Câu nói là: {text}", lang='vi')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")

# --- Giao Diện Streamlit ---
st.set_page_config(page_title="NGÔN NGỮ KÝ HIỆU AI", layout="centered")
st.title("🤟 NGÔN NGỮ KÝ HIỆU AI")

# Sidebar - Cấu hình tìm kiếm
st.sidebar.header("Cài đặt tìm kiếm")
chude_options = ["Tất cả", "đồ dùng học tập", "động vật", "gia đình", "giao thông", "trái cây"]
selected_chude = st.sidebar.selectbox("Chọn chủ đề", chude_options)

# Khu vực nhập liệu
col1, col2 = st.columns([3, 1])
with col1:
    search_query = st.text_input("Nhập câu nói hoặc chữ cái:", "")
with col2:
    search_btn = st.button("Tìm kiếm")

# Khu vực hiển thị Video/Ảnh
display_area = st.empty()

# Xử lý Logic Tìm kiếm
if search_btn and search_query:
    name_search = unidecode(search_query).lower().strip()
    folders = ["video_train", "anh_train", "đồ dùng học tập", "động vật", "gia đình", "giao thông", "trái cây"] if selected_chude == "Tất cả" else [selected_chude]
    
    found = False
    for folder in folders:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                file_name_clean = unidecode(os.path.splitext(file)[0]).lower().strip()
                
                if name_search == file_name_clean:
                    file_path = os.path.join(folder, file)
                    found = True
                    
                    # Phát âm thanh
                    play_audio_st(os.path.splitext(file)[0])
                    
                    # Xử lý Video
                    if file.lower().endswith(('.mp4', '.avi', '.mkv')):
                        cap = cv2.VideoCapture(file_path)
                        st_frame = st.empty() # Placeholder cho video
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret: break
                            
                            # Xử lý và hiển thị
                            processed_frame = process_frame(frame)
                            st_frame.image(processed_frame, channels="RGB", use_container_width=True)
                            time.sleep(0.01) # Giảm tốc độ để giống video thật
                        cap.release()
                        
                    # Xử lý Ảnh
                    elif file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img = Image.open(file_path)
                        st.image(img, caption=file, use_container_width=True)
                    break
        if found: break
    
    if not found:
        st.error("Không tìm thấy ngôn ngữ ký hiệu nào phù hợp!")

# Chức năng chạy file nhận diện ngoài (Subprocess)
st.divider()
if st.button("Chạy Nhận Diện Tay (Cửa sổ riêng)"):
    try:
        # Lưu ý: Subprocess sẽ mở camera trên máy chủ đang chạy script
        import subprocess
        subprocess.Popen(['python', 'nhandientay.py'])
        st.success("Đang khởi động cửa sổ nhận diện...")
    except Exception as e:
        st.error(f"Lỗi: {e}")

# Chức năng Upload Video sẵn có
st.sidebar.divider()
uploaded_file = st.sidebar.file_uploader("Hoặc tải lên video để phân tích", type=["mp4", "avi", "mkv"])
if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame_upload = st.empty()
    
    if st.sidebar.button("Bắt đầu phân tích"):
        play_audio_st(os.path.splitext(uploaded_file.name)[0])
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            processed_frame = process_frame(frame)
            st_frame_upload.image(processed_frame, channels="RGB", use_container_width=True)
        cap.release()
