import streamlit as st
import cv2
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# ======================
# CẤU HÌNH TRANG
# ======================
st.set_page_config(
    page_title="Sign.AI – Ngôn ngữ ký hiệu",
    page_icon="✋",
    layout="centered"
)

st.title("✋ Sign.AI – Nhận diện tay cho người khiếm thính")

# ======================
# KHỞI TẠO MEDIAPIPE
# ======================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ======================
# VIDEO PROCESSOR
# ======================
class HandProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ======================
# GIAO DIỆN
# ======================
st.info("📷 Cho phép trình duyệt dùng camera để bắt khớp tay")

webrtc_streamer(
    key="hand-sign",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=HandProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

st.markdown("""
### ✨ Chức năng hiện tại
- Nhận diện **bàn tay**
- Hiển thị **21 khớp tay**
- Theo dõi **chuyển động realtime**

👉 Có thể mở rộng sang:
- Nhận diện **chữ cái A–Z**
- Dịch **ký hiệu → chữ → giọng nói**
""")
