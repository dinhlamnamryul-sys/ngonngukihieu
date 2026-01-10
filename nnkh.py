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
st.caption("Ứng dụng demo: Bật camera – Bắt khớp tay realtime")

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
            model_complexity=0,  # nhẹ – chạy mượt trên web
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
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ======================
# GIAO DIỆN
# ======================
st.info("📷 Vui lòng cho phép trình duyệt sử dụng camera")

webrtc_streamer(
    key="sign-ai-camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=HandProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,

    # 🔥 FIX LỖI CAMERA – STUN SERVER
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]
    }
)

st.markdown("""
### ✨ Chức năng hiện tại
- ✅ Bật camera Web
- ✅ Nhận diện **21 khớp tay**
- ✅ Theo dõi tay realtime
- ✅ Hoạt động tốt trên **Streamlit Cloud**

### 🚀 Có thể mở rộng
- ✋ Nhận diện chữ cái A–Z
- 🔤 Ghép từ – câu
- 🔊 Phát giọng nói giúp người khiếm thính giao tiếp
- 📚 Thư viện học ngôn ngữ ký hiệu
""")
