import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Hàm nhận diện cử chỉ tay dựa trên hình ảnh ngôn ngữ ký hiệu
# (Lưu ý: Đây là một hàm đơn giản hóa, độ chính xác cần được cải thiện bằng mô hình ML)
def recognize_gesture(landmarks):
    # Lấy tọa độ y của các đầu ngón tay và các đốt ngón tay
    thumb_tip = landmarks[4].y
    index_tip = landmarks[8].y
    middle_tip = landmarks[12].y
    ring_tip = landmarks[16].y
    pinky_tip = landmarks[20].y
    
    thumb_ip = landmarks[3].y
    index_pip = landmarks[6].y
    middle_pip = landmarks[10].y
    ring_pip = landmarks[14].y
    pinky_pip = landmarks[18].y

    # Logic đơn giản để nhận diện một số ký tự cơ bản từ ảnh image_0.png
    # A: Nắm đấm, ngón cái ở bên cạnh
    if index_tip > index_pip and middle_tip > middle_pip and ring_tip > ring_pip and pinky_tip > pinky_pip and landmarks[4].x < landmarks[3].x:
        return "A"
    # B: 4 ngón thẳng, ngón cái gập vào lòng bàn tay
    elif index_tip < index_pip and middle_tip < middle_pip and ring_tip < ring_pip and pinky_tip < pinky_pip and thumb_tip > thumb_ip:
        return "B"
    # C: Bàn tay tạo hình chữ C
    elif index_tip > index_pip and middle_tip > middle_pip and ring_tip > ring_pip and pinky_tip > pinky_pip and landmarks[4].x > landmarks[3].x: # Simplified logic
        return "C"
    # D: Ngón trỏ thẳng, các ngón khác gập
    elif index_tip < index_pip and middle_tip > middle_pip and ring_tip > ring_pip and pinky_tip > pinky_pip:
        return "D"
    # E: Các ngón gập lại, ngón cái gập xuống dưới các ngón khác (như trong image_1.png)
    elif index_tip > index_pip and middle_tip > middle_pip and ring_tip > ring_pip and pinky_tip > pinky_pip and thumb_tip > index_tip:
        return "E"
    # F: Ngón trỏ và ngón cái chạm nhau, 3 ngón còn lại thẳng
    elif abs(index_tip - thumb_tip) < 0.05 and middle_tip < middle_pip and ring_tip < ring_pip and pinky_tip < pinky_pip:
        return "F"
    # V: Ngón trỏ và giữa thẳng, tạo hình chữ V
    elif index_tip < index_pip and middle_tip < middle_pip and ring_tip > ring_pip and pinky_tip > pinky_pip:
        return "V"
    # ... (Cần thêm logic cho các ký tự khác để hoàn thiện bảng chữ cái)
    else:
        return ""

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="Trình thông dịch AI - Ngôn ngữ ký hiệu", layout="wide")

st.markdown("<h1 style='text-align: center; color: white;'>Trình thông dịch AI - Ngôn ngữ ký hiệu</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #a0a0a0;'>HỆ THỐNG ĐANG CHẠY</h3>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("""
        <style>
        .stVideo {
            border-radius: 20px;
            overflow: hidden;
        }
        </style>
        """, unsafe_allow_html=True)
    run = st.checkbox('Bắt đầu Camera')
    FRAME_WINDOW = st.image([])

with col2:
    st.markdown("""
        <style>
        .result-box {
            background-color: #4b50b0;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            color: white;
            font-size: 24px;
            font-weight: bold;
            min-height: 100px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .result-title {
            font-size: 16px;
            color: #a0a0a0;
            margin-bottom: 10px;
        }
        .gesture-result {
            font-size: 48px;
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown('<div class="result-box"><div class="result-title">ĐỘ ỔN ĐỊNH</div><div class="gesture-result" id="gesture-result"></div></div>', unsafe_allow_html=True)

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Không thể truy cập camera.")
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    gesture = ""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = recognize_gesture(hand_landmarks.landmark)
            
            # Cập nhật kết quả lên giao diện
            if gesture:
                st.markdown(f"""
                    <script>
                    document.getElementById("gesture-result").innerText = "{gesture}";
                    </script>
                    """, unsafe_allow_html=True)

    FRAME_WINDOW.image(frame, channels='BGR')

camera.release()
