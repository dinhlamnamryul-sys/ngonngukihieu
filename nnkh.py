import streamlit as st
import cv2
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# =============================
# CẤU HÌNH TRANG
# =============================
st.set_page_config(
    page_title="Sign.AI – Hỗ trợ người khiếm thính",
    page_icon="✋",
    layout="wide"
)

# =============================
# CSS – GIAO DIỆN THÂN THIỆN
# =============================
st.markdown("""
<style>
body {
    background-color: #f8fafc;
}
h1, h2, h3 {
    color: #0f172a;
}
.big-text {
    font-size: 22px;
    font-weight: bold;
}
.card {
    background: white;
    padding: 1.5rem;
    border-radius: 1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# =============================
# MEDIAPIPE
# =============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# =============================
# VIDEO PROCESSOR
# =============================
class HandProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    img,
                    hand,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3),
                    mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =============================
# SIDEBAR – MENU
# =============================
st.sidebar.title("✋ Sign.AI")
menu = st.sidebar.radio(
    "Chức năng",
    [
        "🏠 Trang chủ",
        "✋ Phân tích khớp tay",
        "🤖 AI hiểu cử chỉ (ý tưởng)",
        "📚 Thư viện ký hiệu",
        "🎓 Chế độ học tập"
    ]
)

# =============================
# TRANG CHỦ
# =============================
if menu == "🏠 Trang chủ":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("Sign.AI – Công nghệ vì người khiếm thính")
    st.markdown("""
    <p class="big-text">
    Ứng dụng hỗ trợ người khiếm thính:
    </p>
    <ul class="big-text">
        <li>✋ Nhận diện tay từ camera</li>
        <li>🤖 AI hiểu cử chỉ</li>
        <li>📚 Học ngôn ngữ ký hiệu</li>
        <li>🎓 Luyện tập giao tiếp</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# PHÂN TÍCH KHỚP TAY
# =============================
elif menu == "✋ Phân tích khớp tay":
    st.title("✋ Phân tích khớp tay từ Camera")
    st.info("👉 Giữ tay trước camera – hệ thống sẽ hiển thị 21 khớp tay")

    web
