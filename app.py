import asyncio
import os
import edge_tts
from google import genai
from moviepy.editor import AudioFileClip, CompositeAudioClip, VideoFileClip
import streamlit as st
import whisper

st.set_page_config(
    page_title="Auto Gensub & Dubbing AI", page_icon="🎬", layout="wide"
)

st.title("🎬 Tool Tự Động Dịch & Lồng Tiếng Douyin (100% Free)")
st.caption(
    "Tự động nhận diện tiếng Trung -> AI Dịch Việt bắt trend -> Lồng tiếng &"
    " Ghép Video"
)

# --- CẤU HÌNH SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Cấu Hình AI")

    gemini_api_key = st.text_input(
        "Nhập Gemini API Key (Miễn phí):", type="password"
    )

    st.subheader("🔊 Tùy Chỉnh Âm Thanh")
    voice_option = st.selectbox(
        "Chọn giọng đọc AI:",
        options=["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"],
        format_func=lambda x: (
            "Giọng Nữ truyền cảm (Hoài Mỹ)"
            if "HoaiMy" in x
            else "Giọng Nam tự nhiên (Nam Minh)"
        ),
    )

    speech_rate = st.select_slider(
        "Tốc độ đọc giọng AI (Nên để 0% hoặc +5% để chuẩn khớp video):",
        options=["-10%", "0%", "+5%", "+10%", "+15%", "+20%"],
        value="0%",
    )

    keep_bg_music = st.checkbox("Giữ lại nhạc nền video gốc", value=True)
    bg_vol = st.slider(
        "Âm lượng nhạc nền gốc (%):",
        0,
        50,
        15,
        disabled=not keep_bg_music,
    )


# --- HÀM TÁCH TIẾNG TRUNG BẰNG WHISPER ---
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


def transcribe_chinese(video_path):
    model = load_whisper_model()
    result = model.transcribe(video_path, language="zh")
    return result["text"]


# --- HÀM DỊCH TIẾNG VIỆT CHUẨN GENSUBAI BẰNG GEMINI API ---
def translate_with_gemini(text_zh, api_key):
    client = genai.Client(api_key=api_key.strip())

    prompt = f"""
    Bạn là một chuyên gia biên dịch nội dung video ngắn TikTok/Douyin (Review, Mẹo chỉnh ảnh, Tip/Trick).
    Hãy dịch đoạn thoại tiếng Trung dưới đây sang Tiếng Việt.

    YÊU CẦU BẮT BUỘC:
    1. Văn phong: Cực kỳ tự nhiên, bắt trend, cuốn hút như các reviewer Việt Nam nổi tiếng.
    2. Độ ngắn gọn: Câu dịch phải cực kỳ NGẮN GỌN và SÚC TÍCH để khớp chính xác với thời lượng video ngắn, không dùng từ rườm rà.
    3. Định dạng: Chỉ xuất ra bản dịch Tiếng Việt duy nhất, không ghi thêm bất kỳ lời giải thích nào.

    Thoại tiếng Trung: "{text_zh}"
    """

    for model_name in ["gemini-3.6-flash", "gemini-2.5-flash"]:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            if response and response.text:
                return response.text.strip()
        except Exception:
            continue

    raise Exception(
        "Không thể kết nối tới mô hình Gemini AI. Vui lòng kiểm tra lại API Key!"
    )


# --- HÀM TẠO VOICE & GHÉP VIDEO ---
async def generate_tts(text, voice, rate, output_path):
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(output_path)


def process_video(video_path, audio_path, output_path, keep_bg, bg_volume):
    video = VideoFileClip(video_path)
    vi_voice = AudioFileClip(audio_path)

    # Nếu file âm thanh AI dài hơn video, điều chỉnh cắt hợp lý
    if keep_bg and video.audio is not None:
        bg_audio = video.audio.volumex(bg_volume / 100.0)
        final_audio = CompositeAudioClip([bg_audio, vi_voice])
    else:
        final_audio = vi_voice

    final_video = video.set_audio(final_audio)
    final_video.write_videofile(
        output_path, codec="libx264", audio_codec="aac", fps=video.fps
    )

    video.close()
    vi_voice.close()


# --- GIAO DIỆN CHÍNH ---
uploaded_video = st.file_uploader(
    "Tải video Douyin gốc tiếng Trung lên (.mp4)", type=["mp4", "mov"]
)

if uploaded_video:
    temp_video_path = "temp_input.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.read())

if st.button("🚀 Bắt Đầu Tự Động Dịch & Lồng Tiếng", type="primary"):
    if not uploaded_video:
        st.error("⚠️ Vui lòng tải video lên trước!")
    elif not gemini_api_key:
        st.error(
            "⚠️ Vui lòng nhập Gemini API Key ở thanh bên trái để AI dịch!"
        )
    else:
        temp_audio_path = "temp_voice.mp3"
        output_video_path = "output_dubbed.mp4"

        try:
            with st.spinner(
                "1/3 🎧 Whisper AI đang lắng nghe và bóc thoại Tiếng Trung..."
            ):
                zh_text = transcribe_chinese("temp_input.mp4")
                st.info(f"🗣️ **Thoại tiếng Trung nhận diện được:** {zh_text}")

            with st.spinner(
                "2/3 🤖 Gemini AI đang dịch sang Tiếng Việt bắt trend ngắn gọn..."
            ):
                vi_text = translate_with_gemini(zh_text, gemini_api_key)

            # CHỈNH SỬA NỘI DUNG DỊCH TRỰC TIẾP TRÊN GIAO DIỆN
            st.success("📝 **Bản dịch Tiếng Việt (Có thể chỉnh sửa theo ý bạn bên dưới):**")
            edited_vi_text = st.text_area(
                "Bản dịch Tiếng Việt:", value=vi_text, height=100
            )

            with st.spinner(
                "3/3 🎬 Đang tạo giọng đọc AI & Ghép vào video..."
            ):
                asyncio.run(
                    generate_tts(
                        edited_vi_text, voice_option, speech_rate, temp_audio_path
                    )
                )
                process_video(
                    "temp_input.mp4",
                    temp_audio_path,
                    output_video_path,
                    keep_bg_music,
                    bg_vol,
                )

            st.balloons()
            st.video(output_video_path)

            with open(output_video_path, "rb") as file:
                st.download_button(
                    label="📥 Tải Video Đã Lồng Tiếng Về Máy",
                    data=file,
                    file_name="douyin_dubbed_vietnamese.mp4",
                    mime="video/mp4",
                )

        except Exception as e:
            st.error(f"❌ Xảy ra lỗi trong quá trình xử lý: {str(e)}")

        finally:
            for p in ["temp_input.mp4", "temp_voice.mp3"]:
                if os.path.exists(p):
                    os.remove(p)
