import streamlit as st
import cv2
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import google.generativeai as genai
from PIL import Image

# ======================
# CẤU HÌNH TRANG
# ======================
st.set_page_config(
    page_title="Sign.AI – Ngôn ngữ ký hiệu",
    page_icon="✋",
    layout="centered"
)

st.title("✋ Sign.AI – AI hỗ trợ người khiếm thính")
st.caption("Camera + MediaPipe + Gemini Vision AI")

# ======================
# API KEY
# ======================
api_key = st.secrets.get("GOOGLE_API_KEY", "")

if not api_key:
    st.warning("⚠️ Chưa có Google API Key")
    api_key = st.text_input("Nhập Google API Key:", type="password")

# ======================
# MEDIAPIPE
# ======================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ======================
# AI PHÂN TÍCH ẢNH
# ======================
def analyze_real_image(api_key, image):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    prompt = """
    Đây là hình ảnh bàn tay người.
    Hãy phân tích:
    - Ngón tay nào đang duỗi, ngón nào đang gập
    - Tư thế bàn tay
    - Có thể tương ứng ký hiệu ngôn ngữ tay nào (A, B, C, D, V, I… nếu có)
    Trả lời ngắn gọn, rõ ràng, bằng tiếng Việt.
    """

    response = model.generate_content([prompt, pil_image])
    return response.text


# ======================
# VIDEO PROCESSOR
# ======================
class HandProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        # Lưu frame cho AI
        st.session_state.last_frame = img.copy()

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                    mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ======================
# GIAO DIỆN CAMERA
# ======================
st.info("📷 Cho phép trình duyệt sử dụng camera")

webrtc_streamer(
    key="sign-ai",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=HandProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]
    }
)

# ======================
# NÚT AI PHÂN TÍCH
# ======================
st.divider()

if st.button("🤖 AI phân tích ký hiệu tay"):
    if not api_key:
        st.error("❌ Chưa có Google API Key")
    elif "last_frame" not in st.session_state:
        st.error("❌ Chưa có hình ảnh từ camera")
    else:
        with st.spinner("AI đang phân tích cử chỉ tay..."):
            result = analyze_real_image(
                api_key,
                st.session_state.last_frame
            )
        st.success("✅ Kết quả AI:")
        st.write(result)

# ======================
# THÔNG TIN
# ======================
st.markdown("""
### ✨ Chức năng
- ✅ Camera realtime
- ✅ Bắt **21 khớp tay**
- ✅ AI hiểu **cử chỉ bàn tay**
- ✅ Hỗ trợ **người khiếm thính giao tiếp**

### 🚀 Có thể mở rộng
- Nhận diện chữ cái A–Z
- Ghép từ → câu
- Text → Speech cho người nghe
- Chế độ học tập cho HS khiếm thính
""")
