import streamlit as st
import cv2
import mediapipe as mp
import math
import numpy as np
import av  # Thư viện xử lý video frame quan trọng
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import time
from datetime import datetime

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Sign.AI - Dịch thuật Ngôn ngữ Ký hiệu",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- KHỞI TẠO SESSION STATE (Giả lập Database) ---
if 'user' not in st.session_state:
    st.session_state.user = {'name': 'Anonymous', 'role': 'user'}
if 'media_bank' not in st.session_state:
    st.session_state.media_bank = []
if 'detected_text' not in st.session_state:
    st.session_state.detected_text = ""

# --- CSS TÙY CHỈNH ---
st.markdown("""
<style>
    .main { background-color: #fcfdfe; color: #0f172a; }
    [data-testid="stSidebar"] { background-color: #0f172a; color: white; }
    div.stButton > button {
        border-radius: 1rem; font-weight: bold; text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# --- LOGIC XỬ LÝ ẢNH (MEDIAPIPE) ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandGestureProcessor(VideoProcessorBase):
    def __init__(self):
        # Khởi tạo model MediaPipe
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.last_letter = ""
        self.stability_counter = 0

    def analyze_hand_gesture(self, landmarks):
        lm = landmarks.landmark
        
        def dist(p1, p2):
            return math.sqrt((lm[p1].x - lm[p2].x)**2 + (lm[p1].y - lm[p2].y)**2)
        
        def is_ext(tip, pip):
            return lm[tip].y < lm[pip].y

        # Logic nhận diện đơn giản hóa
        thumb_ext = lm[4].x < lm[3].x
        index_ext = is_ext(8, 6)
        middle_ext = is_ext(12, 10)
        ring_ext = is_ext(16, 14)
        pinky_ext = is_ext(20, 18)

        # Logic mapping (như code cũ)
        if not index_ext and not middle_ext and not ring_ext and not pinky_ext and lm[4].y < lm[6].y: return "A"
        if index_ext and middle_ext and ring_ext and pinky_ext and dist(8, 12) < 0.04: return "B"
        if index_ext and not middle_ext and not ring_ext and not pinky_ext: return "D" # Ví dụ rút gọn
        if pinky_ext and not index_ext: return "I"
        if index_ext and middle_ext and not ring_ext: return "V"
        if dist(8, 4) < 0.05 and dist(12, 4) < 0.05: return "O"
        
        return "" # Trả về rỗng nếu không khớp

    def recv(self, frame):
        # 1. Chuyển đổi từ av.VideoFrame sang numpy array (OpenCV format)
        img = frame.to_ndarray(format="bgr24")
        
        # 2. Xử lý ảnh (Mirror -> RGB -> MediaPipe)
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        detected_char = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Vẽ khung xương tay
                mp_drawing.draw_landmarks(
                    img, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                )
                
                # Phân tích cử chỉ
                detected_char = self.analyze_hand_gesture(hand_landmarks)
                
                if detected_char:
                    # Logic ổn định (Debounce)
                    if detected_char == self.last_letter:
                        self.stability_counter += 1
                    else:
                        self.last_letter = detected_char
                        self.stability_counter = 0

                    # Hiển thị chữ lên màn hình
                    cv2.putText(img, f"Ky tu: {detected_char}", (30, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4, cv2.LINE_AA)
                    
                    # Thanh tiến trình xác nhận
                    bar_width = int((self.stability_counter / 20) * 200)
                    cv2.rectangle(img, (30, 100), (30 + bar_width, 120), (0, 255, 0), -1)

        # 3. QUAN TRỌNG: Chuyển đổi ngược từ numpy array về av.VideoFrame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- DANH MỤC ---
CATEGORIES = [
    {"id": "all", "label": "Tất cả"},
    {"id": "alphabet", "label": "Bảng chữ cái"},
    {"id": "communication", "label": "Giao tiếp cơ bản"},
    {"id": "school", "label": "Đồ dùng học tập"},
]

# --- UI CHÍNH ---
with st.sidebar:
    st.title("✋ Sign.AI")
    menu = st.radio("Menu", ["Dịch thuật AI", "Thư viện Ký hiệu", "Quản trị Admin"])

if menu == "Dịch thuật AI":
    st.header("Dịch Thuật Camera")
    st.write("Bật camera và đưa tay vào khung hình.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # WebRTC Streamer
        webrtc_streamer(
            key="sign-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=HandGestureProcessor, # Sử dụng class xử lý mới
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col2:
        st.info("💡 Hướng dẫn: Giữ yên tay khoảng 1 giây để hệ thống chốt chữ cái.")

elif menu == "Thư viện Ký hiệu":
    st.header("Thư viện")
    search = st.text_input("Tìm kiếm")
    
    if st.session_state.media_bank:
        for item in st.session_state.media_bank:
            if search.lower() in item['name'].lower():
                with st.expander(f"{item['name']} ({item['category']})"):
                    if item['type'] == 'image':
                        st.image(item['data'])
                    else:
                        st.video(item['data'])
    else:
        st.warning("Chưa có dữ liệu. Hãy vào Admin để thêm.")

elif menu == "Quản trị Admin":
    st.header("Upload Dữ Liệu")
    
    with st.form("upload"):
        name = st.text_input("Tên ký hiệu")
        cat = st.selectbox("Danh mục", [c['label'] for c in CATEGORIES])
        file = st.file_uploader("File ảnh/video")
        submit = st.form_submit_button("Lưu")
        
        if submit and file and name:
            ftype = 'video' if 'video' in file.type else 'image'
            st.session_state.media_bank.append({
                "name": name,
                "category": cat,
                "type": ftype,
                "data": file.read()
            })
            st.success("Đã lưu thành công!")
