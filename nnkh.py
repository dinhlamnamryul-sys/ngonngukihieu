import streamlit as st
import google.generativeai as genai

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="Gemini Chatbot - BYOK",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Chatbot Gemini (Tự nhập Key)")

# --- 2. THANH BÊN (SIDEBAR) ĐỂ NHẬP KEY ---
with st.sidebar:
    st.header("🔑 Cấu hình API Key")
    
    # Ô nhập liệu (type="password" để ẩn ký tự thành dấu chấm tròn)
    user_api_key = st.text_input(
        "Nhập Google API Key của bạn:",
        type="password",
        placeholder="Dán key bắt đầu bằng AIza... vào đây"
    )
    
    # Hướng dẫn lấy key
    st.markdown("---")
    st.markdown(
        "Chưa có Key? [Lấy miễn phí tại Google AI Studio](https://aistudio.google.com/app/apikey)"
    )
    
    # Nút xóa lịch sử chat
    if st.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        st.rerun()

# --- 3. KIỂM TRA KEY ---
if not user_api_key:
    # Nếu chưa nhập Key thì hiện thông báo và dừng chương trình
    st.info("👈 Vui lòng nhập API Key của bạn ở thanh bên trái để bắt đầu trò chuyện.")
    st.stop() # Dừng code tại đây, không chạy phần dưới cho đến khi có Key

# --- 4. CẤU HÌNH MODEL ---
try:
    genai.configure(api_key=user_api_key)
    
    # Cấu hình tính cách model
    system_instruction = "Bạn là trợ lý ảo hữu ích, trả lời ngắn gọn và đi thẳng vào vấn đề."
    
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        system_instruction=system_instruction
    )
except Exception as e:
    st.error(f"API Key không hợp lệ. Vui lòng kiểm tra lại. Lỗi: {e}")
    st.stop()

# --- 5. GIAO DIỆN CHAT (GIỐNG CŨ) ---

# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị tin nhắn cũ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý nhập liệu mới
if prompt := st.chat_input("Hỏi tôi bất cứ điều gì..."):
    # Hiển thị câu hỏi người dùng
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # AI trả lời
    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            try:
                # Gọi API
                response = model.generate_content(prompt)
                
                # Hiển thị và lưu kết quả
                st.markdown(response.text)
                st.session_state.messages.append({"role": "model", "content": response.text})
            
            except Exception as e:
                st.error(f"Có lỗi xảy ra (có thể do Key sai hoặc mạng lỗi): {e}")
