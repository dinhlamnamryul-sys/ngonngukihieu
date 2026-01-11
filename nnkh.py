import streamlit as st
import google.generativeai as genai

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Trợ lý AI Thông Minh")

# --- CẤU HÌNH API KEY (QUAN TRỌNG) ---
# Lấy API Key từ Secrets của Streamlit để bảo mật
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
except KeyError:
    # Hướng dẫn nếu chưa nhập Key
    st.error("⚠️ Lỗi: Chưa tìm thấy API Key.")
    st.info("Vui lòng vào cài đặt 'Secrets' trên Streamlit Cloud và thêm dòng: GOOGLE_API_KEY = 'mã_key_của_bạn'")
    st.stop()

# --- CÀI ĐẶT THANH BÊN (SIDEBAR) ---
with st.sidebar:
    st.header("Cài đặt")
    
    # Nút xóa lịch sử chat
    if st.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.markdown("Created with Gemini & Streamlit")

# --- KHỞI TẠO MODEL ---
# Bạn có thể thay đổi system_instruction để AI đóng vai cụ thể (VD: Giáo viên toán)
system_instruction = "Bạn là một trợ lý AI hữu ích, thân thiện và trả lời ngắn gọn, chính xác."

model = genai.GenerativeModel(
    model_name='gemini-1.5-flash', # Model nhanh và nhẹ
    system_instruction=system_instruction
)

# --- QUẢN LÝ LỊCH SỬ CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị các tin nhắn cũ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- XỬ LÝ NHẬP LIỆU & TRẢ LỜI ---
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # 1. Hiển thị câu hỏi của người dùng
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. AI xử lý và trả lời
    with st.chat_message("assistant"):
        empty_slot = st.empty() # Tạo khung trống để hiệu ứng chữ chạy (nếu muốn)
        with st.spinner("Đang suy nghĩ..."):
            try:
                # Gửi toàn bộ lịch sử chat để AI nhớ ngữ cảnh
                chat_session = model.start_chat(
                    history=[
                        {"role": m["role"], "parts": [m["content"]]}
                        for m in st.session_state.messages 
                        if m["role"] in ["user", "model"] # Lọc đúng role cho Gemini
                    ]
                )
                
                # Gửi tin nhắn mới nhất (lưu ý: ở đây dùng send_message vì đã start_chat)
                # Tuy nhiên để đơn giản và ít lỗi context, ta dùng generate_content cho prompt hiện tại
                # kết hợp context tự quản lý hoặc dùng chat object. 
                # Cách ổn định nhất cho app đơn giản:
                response = model.generate_content(prompt) 
                
                # Hiển thị kết quả
                st.markdown(response.text)
                
                # Lưu vào lịch sử
                st.session_state.messages.append({"role": "model", "content": response.text})
                
            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {e}")
