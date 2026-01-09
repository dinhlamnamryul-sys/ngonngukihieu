import streamlit as st
import cv2
import mediapipe as mp
import math
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import time
from datetime import datetime

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Sign.AI - Dịch thuật Ngôn ngữ Ký hiệu",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GIẢ LẬP FIREBASE (SESSION STATE) ---
# Trong thực tế, bạn sẽ thay thế phần này bằng firebase-admin sdk
if 'user' not in st.session_state:
    st.session_state.user = {'name': 'Anonymous', 'role': 'user'}
if 'media_bank' not in st.session_state:
    st.session_state.media_bank = []
if 'detected_text' not in st.session_state:
    st.session_state.detected_text = ""

# --- CSS TÙY CHỈNH ĐỂ GIỐNG GIAO DIỆN REACT CŨ ---
st.markdown("""
<style>
    /* Tổng thể */
    .main {
        background-color: #fcfdfe;
        color: #0f172a;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        color: white;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] span {
        color: white;
    }
    /* Cards */
    div.css-1r6slb0 {
        background-color: white;
        border-radius: 2rem;
        padding: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #f1f5f9;
    }
    /* Buttons */
    .stButton>button {
        border-radius: 1rem;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
</style>
""", unsafe_allow_html=True)

# --- LOGIC NHẬN DIỆN (MEDIAPIPE) ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandGestureProcessor(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.last_letter = ""
        self.stability_counter = 0
        self.locked = False
        self.lock_time = 0

    def analyze_hand_gesture(self, landmarks):
        lm = landmarks.landmark
        
        # Hàm phụ trợ tính khoảng cách và trạng thái ngón tay
        def dist(p1, p2):
            return math.sqrt((lm[p1].x - lm[p2].x)**2 + (lm[p1].y - lm[p2].y)**2)
        
        def is_ext(tip, pip):
            return lm[tip].y < lm[pip].y

        thumb_ext = lm[4].x < lm[3].x  # Giả sử tay phải, cần mirror ảnh
        index_ext = is_ext(8, 6)
        middle_ext = is_ext(12, 10)
        ring_ext = is_ext(16, 14)
        pinky_ext = is_ext(20, 18)

        # Logic dịch từ code React sang Python
        if not index_ext and not middle_ext and not ring_ext and not pinky_ext and lm[4].y < lm[6].y: return "A"
        if not index_ext and not middle_ext and not ring_ext and not pinky_ext and lm[4].y < lm[5].y and lm[4].x > lm[3].x: return "Ă"
        if not index_ext and not middle_ext and not ring_ext and not pinky_ext and dist(4, 10) < 0.05: return "Â"
        if index_ext and middle_ext and ring_ext and pinky_ext and dist(8, 12) < 0.04: return "B"
        if not index_ext and not middle_ext and dist(8, 4) > 0.12 and lm[8].y > lm[5].y and lm[4].y > lm[17].y: return "C"
        if index_ext and not middle_ext and not ring_ext and not pinky_ext and dist(12, 4) < 0.05: return "D"
        if index_ext and not middle_ext and not ring_ext and dist(4, 9) < 0.05: return "Đ"
        if not index_ext and not middle_ext and not ring_ext and not pinky_ext and lm[8].y > lm[7].y and lm[12].y > lm[11].y: return "E"
        if not index_ext and not middle_ext and not ring_ext and dist(4, 10) < 0.04: return "Ê"
        if lm[8].x < lm[6].x and lm[4].x < lm[3].x and not middle_ext: return "G"
        if lm[8].x < lm[6].x and lm[12].x < lm[10].x and not ring_ext: return "H"
        if pinky_ext and not index_ext and not middle_ext and not ring_ext: return "I"
        if index_ext and middle_ext and dist(4, 9) < 0.06: return "K"
        if index_ext and not middle_ext and not ring_ext and thumb_ext: return "L"
        if not index_ext and not middle_ext and not ring_ext and dist(4, 17) < 0.08: return "M"
        if not index_ext and not middle_ext and ring_ext and dist(4, 13) < 0.08: return "N"
        if dist(8, 4) < 0.04 and dist(12, 4) < 0.04 and dist(16, 4) < 0.04 and dist(20, 4) < 0.04: return "O"
        if dist(8, 4) < 0.04 and dist(12, 4) < 0.04 and lm[0].y > 0.8: return "Ô"
        if dist(8, 4) < 0.04 and dist(12, 4) < 0.04 and pinky_ext: return "Ơ"
        if lm[8].y > lm[6].y and lm[12].y > lm[10].y and dist(4, 9) < 0.06: return "P"
        if lm[8].y > lm[6].y and lm[4].y > lm[3].y and not middle_ext: return "Q"
        if index_ext and middle_ext and lm[8].x > lm[12].x: return "R"
        if not index_ext and not middle_ext and not ring_ext and not pinky_ext and dist(4, 10) < 0.05: return "S"
        if not index_ext and not middle_ext and not ring_ext and dist(4, 7) < 0.04: return "T"
        if index_ext and middle_ext and not ring_ext and dist(8, 12) < 0.03: return "U"
        if index_ext and middle_ext and pinky_ext: return "Ư"
        if index_ext and middle_ext and not ring_ext and dist(8, 12) > 0.1: return "V"
        if index_ext and middle_ext and ring_ext and not pinky_ext: return "W"
        if not index_ext and lm[8].y < lm[5].y and not middle_ext: return "X"
        if thumb_ext and pinky_ext and not index_ext and not middle_ext and not ring_ext: return "Y"
        if index_ext and not middle_ext and not ring_ext and not pinky_ext and not thumb_ext: return "Z"

        return None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror image như React code
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(img_rgb)
        detected_char = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                detected_char = self.analyze_hand_gesture(hand_landmarks)
                
                if detected_char:
                    # Logic ổn định nhận diện (Debounce)
                    if detected_char == self.last_letter:
                        self.stability_counter += 1
                    else:
                        self.last_letter = detected_char
                        self.stability_counter = 0

                    # Hiển thị chữ lên màn hình video
                    cv2.putText(img, detected_char, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                                3, (255, 0, 0), 5, cv2.LINE_AA)
                    
                    # Thanh tiến trình ổn định
                    bar_width = int((self.stability_counter / 20) * 200)
                    cv2.rectangle(img, (50, 120), (50 + bar_width, 140), (0, 255, 0), -1)

        return img

# --- DANH MỤC ---
CATEGORIES = [
    {"id": "all", "label": "Tất cả", "icon": "LayoutGrid"},
    {"id": "alphabet", "label": "Bảng chữ cái", "icon": "Type"},
    {"id": "communication", "label": "Giao tiếp cơ bản", "icon": "MessageSquare"},
    {"id": "school", "label": "Đồ dùng học tập", "icon": "Pencil"},
    {"id": "fruits", "label": "Các loại quả", "icon": "Apple"},
    {"id": "traffic", "label": "Giao thông", "icon": "Car"},
    {"id": "animals", "label": "Động vật", "icon": "Book"},
    {"id": "family", "label": "Gia đình", "icon": "Users"},
]

# --- GIAO DIỆN CHÍNH ---

# Sidebar
with st.sidebar:
    st.title("✋ Sign.AI")
    st.write("---")
    menu = st.radio(
        "Menu",
        ["Dịch thuật AI", "Thư viện Ký hiệu", "Quản trị Admin"],
        captions=["Nhận diện qua Camera", "Tra cứu từ vựng", "Upload dữ liệu"]
    )
    st.write("---")
    st.info("Phiên bản Python Streamlit")

# --- TAB 1: DỊCH THUẬT AI ---
if menu == "Dịch thuật AI":
    st.header("Dịch Thuật Toàn Diện")
    st.caption("Nhận diện chữ cái tiếng Việt dựa trên khớp tay (MediaPipe Python).")

    col1, col2 = st.columns([3, 1])

    with col1:
        # Camera Stream
        webrtc_streamer(
            key="sign-detection",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=HandGestureProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
    with col2:
        st.subheader("Kết quả")
        # Lưu ý: Trong Streamlit WebRTC, việc truyền dữ liệu từ thread video về UI chính rất phức tạp.
        # Ở bản demo này, kết quả nhận diện được vẽ trực tiếp lên video stream.
        st.info("Hướng camera vào tay bạn. Chữ cái sẽ hiện trên video khi nhận diện ổn định.")
        
        if st.button("Xóa văn bản", type="primary"):
            st.session_state.detected_text = ""
            st.toast("Đã xóa văn bản!")

    st.markdown("---")
    st.warning("Lưu ý: Để có kết quả tốt nhất, hãy đảm bảo ánh sáng tốt và đưa tay trọn vào khung hình.")

# --- TAB 2: THƯ VIỆN KÝ HIỆU ---
elif menu == "Thư viện Ký hiệu":
    st.header("Thư viện Ký hiệu")
    
    # Thanh tìm kiếm và bộ lọc
    col_search, col_filter = st.columns([2, 1])
    with col_search:
        search_query = st.text_input("🔍 Tìm kiếm", placeholder="Nhập tên ký hiệu...")
    with col_filter:
        selected_cat = st.selectbox("Danh mục", [c["label"] for c in CATEGORIES])

    # Lọc dữ liệu
    filtered_data = [
        item for item in st.session_state.media_bank 
        if (search_query.lower() in item['name'].lower()) and 
           (selected_cat == "Tất cả" or item['category'] == selected_cat)
    ]

    if not filtered_data:
        st.info("Chưa có dữ liệu nào. Hãy sang tab Admin để thêm dữ liệu!")
    else:
        # Hiển thị Grid
        cols = st.columns(4)
        for idx, item in enumerate(filtered_data):
            with cols[idx % 4]:
                with st.container():
                    st.write(f"**{item['name']}**")
                    st.caption(item['category'])
                    if item['type'] == 'image':
                        st.image(item['data'], use_column_width=True)
                    elif item['type'] == 'video':
                        st.video(item['data'])
                    st.divider()

# --- TAB 3: QUẢN TRỊ ADMIN ---
elif menu == "Quản trị Admin":
    st.header("Admin Sign Bank")
    st.caption("Cập nhật dữ liệu mẫu (Giả lập Firebase).")

    col_form, col_list = st.columns([1, 2])

    with col_form:
        st.subheader("Tải lên")
        with st.form("upload_form"):
            cat_input = st.selectbox("Danh mục", [c["label"] for c in CATEGORIES if c["id"] != "all"])
            uploaded_files = st.file_uploader("Chọn ảnh/video", type=['png', 'jpg', 'mp4'], accept_multiple_files=True)
            
            submitted = st.form_submit_button("Xác nhận tải lên")
            
            if submitted and uploaded_files:
                for file in uploaded_files:
                    file_type = 'video' if 'video' in file.type else 'image'
                    # Lưu vào Session State (Bộ nhớ tạm)
                    new_item = {
                        "id": str(int(time.time() * 1000)),
                        "name": file.name.split('.')[0],
                        "category": cat_input,
                        "type": file_type,
                        "data": file.read(), # Lưu binary data
                        "createdAt": datetime.now()
                    }
                    st.session_state.media_bank.append(new_item)
                st.success(f"Đã thêm {len(uploaded_files)} mục!")

    with col_list:
        st.subheader("Dữ liệu hiện có")
        if st.session_state.media_bank:
            df_display = [
                {"Tên": i["name"], "Danh mục": i["category"], "Loại": i["type"]} 
                for i in st.session_state.media_bank
            ]
            st.dataframe(df_display, use_container_width=True)
            
            if st.button("Xóa toàn bộ dữ liệu (Reset)"):
                st.session_state.media_bank = []
                st.rerun()
        else:
            st.info("Ngân hàng dữ liệu trống.")
