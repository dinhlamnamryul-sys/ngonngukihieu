import streamlit as st
import cv2
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# ================== CẤU HÌNH TRANG ==================
st.set_page_config(
    page_title="Hệ thống NNKH AI",
    page_icon="👐",
    layout="wide"
)

# ================== DỮ LIỆU GIẢ LẬP ==================
mock_library = [
    {"id": 1, "name": "Xin chào", "category": "Gia đình", "type": "video",
     "url": "https://www.w3schools.com/html/mov_bbb.mp4"},
    {"id": 2, "name": "Quả táo", "category": "Trái cây", "type": "image",
     "url": "https://images.unsplash.com/photo-1560806887-1e4cd0b6bcd6?w=400"},
    {"id": 3, "name": "Bút chì", "category": "Đồ dùng học tập", "type": "image",
     "url": "https://images.unsplash.com/photo-1512036667332-2323862660f9?w=400"},
    {"id": 4, "name": "Con mèo", "category": "Động vật", "type": "video",
     "url": "https://www.w3schools.com/html/movie.mp4"},
    {"id": 5, "name": "Ô tô", "category": "Giao thông", "type": "image",
     "url": "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=400"},
]

categories = ["Tất cả", "Gia đình", "Trái cây", "Đồ dùng học tập", "Động vật", "Giao thông"]

# ================== AI MEDIAPIPE ==================
class HandDetectorProcessor(VideoTransformerBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                )

            cv2.putText(
                img, "DANG NHAN DIEN TAY",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2
            )
        else:
            cv2.putText(
                img, "MOI GIO TAY LEN",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2
            )

        return img

# ================== GIAO DIỆN ==================
st.title("👐 Hệ thống Ngôn ngữ Ký hiệu AI")

tab1, tab2 = st.tabs(["📚 Thư viện học tập", "📷 Nhận diện AI"])

# ================== TAB 1 ==================
with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🔍 Tìm kiếm ký hiệu")
        keyword = st.text_input("Nhập tên ký hiệu:")
        category = st.selectbox("Danh mục", categories)
        search_btn = st.button("Tra cứu")

    with col2:
        if search_btn or keyword:
            results = mock_library

            if keyword:
                results = [i for i in results if keyword.lower() in i["name"].lower()]

            if category != "Tất cả":
                results = [i for i in results if i["category"] == category]

            if results:
                res = results[0]
                st.success(f"Kết quả: {res['name']}")
                if res["type"] == "video":
                    st.video(res["url"])
                else:
                    st.image(res["url"], use_container_width=True)
            else:
                st.error("❌ Không tìm thấy ký hiệu.")
        else:
            st.info("👉 Nhập từ khóa để tra cứu ký hiệu.")

# ================== TAB 2 ==================
with tab2:
    st.subheader("📷 Nhận diện cử chỉ tay thời gian thực")
    st.write("Nhấn **Start** và cho phép truy cập Camera.")

    webrtc_streamer(
        key="hand-sign-detect",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=HandDetectorProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# ================== FOOTER ==================
st.markdown("---")
st.caption("© 2026 | Streamlit – MediaPipe – AI")
