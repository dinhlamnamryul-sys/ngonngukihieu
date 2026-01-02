
import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO
import os
from unidecode import unidecode

# =====================
# CẤU HÌNH STREAMLIT
# =====================
st.set_page_config(
    page_title="Ngôn Ngữ Ký Hiệu Từ Ảnh",
    page_icon="🤟",
    layout="wide"
)

st.title("🤟 CHUYỂN ĐỀ TOÁN → NGÔN NGỮ KÝ HIỆU (VIỆT – H’MÔNG)")

# =====================
# HƯỚNG DẪN API KEY
# =====================
with st.expander("🔑 Hướng dẫn lấy Google API Key"):
    st.markdown("""
1. Vào: https://aistudio.google.com/app/apikey  
2. Đăng nhập Gmail  
3. Nhấn **Create API Key**  
4. Copy và dán vào bên dưới  

⚠️ Không chia sẻ API Key
""")

api_key = st.text_input("🔐 Google API Key", type="password")

if not api_key:
    st.warning("⚠️ Cần nhập API Key để sử dụng AI")
    st.stop()

# =====================
# HÀM GỌI GEMINI
# =====================
def analyze_image_with_gemini(api_key, image, prompt):
    if image.mode == "RGBA":
        image = image.convert("RGB")

    buf = BytesIO()
    image.save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    MODEL = "gemini-2.5-flash"
    URL = f"https://generativelanguage.googleapis.com/v1/models/{MODEL}:generateContent?key={api_key}"

    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img_b64
                    }
                }
            ]
        }]
    }

    res = requests.post(URL, json=payload)
    if res.status_code != 200:
        return f"❌ Lỗi API {res.status_code}: {res.text}"

    data = res.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return "❌ Không nhận được nội dung từ AI."

# =====================
# PROMPT NGÔN NGỮ KÝ HIỆU
# =====================
PROMPT_NNKH = """
Bạn là CHUYÊN GIA NGÔN NGỮ KÝ HIỆU VIỆT NAM (VSL) cho người khiếm thính.

NHIỆM VỤ:
- Phân tích bài toán trong ảnh.
- KHÔNG giải theo văn nói.
- CHUYỂN TOÀN BỘ nội dung sang NGÔN NGỮ KÝ HIỆU.

=================================
QUY TẮC BẮT BUỘC
=================================
- Không văn dài.
- Không kể chuyện.
- Dùng TỪ KHÓA – ĐỘNG TÁC – THỨ TỰ KÝ HIỆU.
- Mỗi dòng = 1 ý.
- TỪ KÝ HIỆU viết IN HOA.
- Công thức toán đặt trong $$ $$.
- Không sinh ký tự lạ.

=================================
1️⃣ PHÂN TÍCH ĐỀ (KÝ HIỆU)
=================================
- Dòng 1: Ký hiệu (VIỆT – IN HOA).
- Dòng 2: Ký hiệu (H’MÔNG – IN HOA).
- Dòng 3: Thứ tự ký hiệu (→).

=================================
2️⃣ GIẢI BÀI BẰNG KÝ HIỆU
=================================
Mỗi bước gồm 3 dòng:
- VIỆT (KÝ HIỆU).
- H’MÔNG (KÝ HIỆU).
- CÔNG THỨC LaTeX sạch.

=================================
3️⃣ DANH SÁCH TỪ CẦN VIDEO KÝ HIỆU
=================================
- Mỗi dòng 1 từ IN HOA.
- Không giải thích thêm.
"""

# =====================
# NHẬP ẢNH
# =====================
st.subheader("📷 Chụp hoặc tải ảnh đề bài")

col1, col2 = st.columns(2)

with col1:
    cam = st.camera_input("Chụp ảnh")

with col2:
    upload = st.file_uploader("Tải ảnh", type=["jpg", "png", "jpeg"])

image = None
if cam:
    image = Image.open(cam)
elif upload:
    image = Image.open(upload)

# =====================
# XỬ LÝ
# =====================
if image:
    st.image(image, caption="Ảnh đề bài", use_column_width=True)

    if st.button("🤖 CHUYỂN SANG NGÔN NGỮ KÝ HIỆU", type="primary"):
        with st.spinner("⏳ AI đang phân tích & chuyển sang ký hiệu..."):
            result = analyze_image_with_gemini(api_key, image, PROMPT_NNKH)

        if result.startswith("❌"):
            st.error(result)
        else:
            st.success("✅ Hoàn thành")
            st.markdown(result)
