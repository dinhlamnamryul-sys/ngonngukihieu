import streamlit as st
import cv2
import mediapipe as mp
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import threading

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="SIGN.AI - Trình thông dịch", page_icon="✋", layout="wide")

# --- 2. CSS ĐỂ TẠO GIAO DIỆN GIỐNG ẢNH ---
st.markdown("""
    <style>
    /* Tổng thể nền tối */
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b2c;
        border-right: 1px solid #2d2d44;
    }
    
    /* Header chính */
    .main-header {
        font-size: 32px;
        font-weight: bold;
        color: white;
        margin-bottom: 10px;
    }
    
    /* Badge trạng thái */
    .status-badge {
        background-color: #4b50b0;
        color: white;
        padding: 5px 15px;
        border-radius: 15px;
        font-size: 14px;
        font-weight: bold;
        float: right;
    }

    /* Box kết quả (Giống hình E tím) */
    .result-box-container {
        background-color: #1c2136;
        border: 1px solid #2d2d44;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
    }
    
    .result-label {
        color: #8b8da0;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    
    .result-value {
        background-color: #5865F2; /* Màu tím xanh giống Discord/Ảnh mẫu */
        color: white;
        font-size: 60px;
        font-weight: bold;
        border-radius: 12px;
        padding: 20px;
        display: inline-block;
        min-width: 100px;
        box-shadow: 0 4px 15px rgba(88, 101, 242, 0.4);
    }

    /* Tùy chỉnh nút bấm camera */
    button {
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. XỬ LÝ MEDIAPIPE VÀ LOGIC NHẬN DIỆN ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Biến toàn cục để lưu kết quả nhận diện (Thread safe)
lock = threading.Lock()
shared_state = {"prediction": "..."}

def recognize_gesture(landmarks):
    """
    Hàm nhận diện cử chỉ dựa trên tọa độ (Rule-based đơn giản).
    Cần training model AI thực sự để nhận diện chính xác 26 chữ cái.
    Đây là logic demo cho các chữ cái cơ bản trong ảnh mẫu.
    """
    thumb_tip = landmarks[4].y
    index_tip = landmarks[8].y
    middle_tip = landmarks[12].y
    ring_tip = landmarks[16].y
    pinky_tip = landmarks[20].y
    
    thumb_ip = landmarks[3].y
    index_pip = landmarks[6].y
    
    # Logic nhận diện (Ví dụ)
    # A: Nắm đấm, ngón cái áp sát cạnh
    if index_tip > index_pip and middle_tip > landmarks[10].y and ring_tip > landmarks[14].y and pinky_tip > landmarks[18].y:
        return "A"
    
    # B: 4 ngón thẳng, ngón cái gập (Bàn tay mở)
    if index_tip < index_pip and middle_tip < landmarks[10].y and ring_tip < landmarks[14].y and pinky_tip < landmarks[18].y and thumb_tip > thumb_ip:
        return "B"

    # V: Ngón trỏ và giữa tạo chữ V
    if index_tip < index_pip and middle_tip < landmarks[10].y and ring_tip > landmarks[14].y and pinky_tip > landmarks[18].y:
        return "V"
        
    # L: Ngón cái và trỏ vuông góc
    if thumb_tip < landmarks[3].y and index_tip < index_pip and middle_tip > landmarks[10].y:
        return "L"

    # E: (Giống hình mẫu) Các ngón co lại, ngón cái gập dưới
    # Logic: Các đầu ngón tay đều thấp (tọa độ y cao) gần gò bàn tay
    if index_tip > index_pip and middle_tip > landmarks[10].y and ring_tip > landmarks[14].y and pinky_tip > landmarks[18].y and thumb_tip > index_tip:
        return "E"

    return ""

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Lật ảnh để giống gương
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(img_rgb)
        
        gesture = ""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Vẽ khung xương tay
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Nhận diện
                gesture = recognize_gesture(hand_landmarks.landmark)
                
                # Cập nhật kết quả vào biến chung
                with lock:
                    shared_state["prediction"] = gesture if gesture else "..."
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. BỐ CỤC GIAO DIỆN (LAYOUT) ---

# Sidebar
with st.sidebar:
    st.title("🖐️ SIGN.AI")
    st.markdown("---")
    st.info("💡 Hướng dẫn: Đưa tay vào khung hình camera để nhận diện chữ cái.")
    st.markdown("### Thư viện ký hiệu")
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/47/American_Sign_Language_ASL.svg", caption="Bảng chữ cái tham khảo")
    
    st.markdown("---")
    st.caption("Developed by Gemini User")

# Main Content
col1, col2 = st.columns([3, 1.5])

with col1:
    st.markdown('<div class="main-header">Trình thông dịch AI <span class="status-badge">HỆ THỐNG ĐANG CHẠY</span></div>', unsafe_allow_html=True)
    
    # Khu vực Camera WebRTC
    ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.markdown("<br><br>", unsafe_allow_html=True) # Spacer
    
    # Hộp hiển thị kết quả Real-time
    st.markdown("""
        <div class="result-box-container">
            <div class="result-label">KẾT QUẢ NHẬN DIỆN</div>
            <div class="result-value" id="prediction-placeholder">?</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Phần hiển thị văn bản thô
    st.markdown("<br>", unsafe_allow_html=True)
    st.text_area("Văn bản thô (Ghi chú):", height=150, placeholder="Các ký tự sẽ xuất hiện tại đây...")

# --- 5. CƠ CHẾ CẬP NHẬT KẾT QUẢ TỪ THREAD WEBRTC RA UI ---
# Streamlit cần reload để update UI, dùng st_autorefresh hoặc placeholder loop
import time
placeholder = st.empty()

if ctx.state.playing:
    while True:
        with lock:
            current_pred = shared_state["prediction"]
        
        # Cập nhật trực tiếp vào hộp HTML bên phải bằng JavaScript hack hoặc hiển thị lại
        # Vì Streamlit chặn JS trực tiếp, ta dùng markdown đè lên vùng đó
        with col2:
             st.markdown(f"""
                <div class="result-box-container">
                    <div class="result-label">ĐỘ ỔN ĐỊNH: CAO</div>
                    <div class="result-value">{current_pred}</div>
                </div>
            """, unsafe_allow_html=True)
        
        time.sleep(0.1) # Cập nhật mỗi 0.1s
