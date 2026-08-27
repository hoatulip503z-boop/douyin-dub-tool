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
        options=["vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"],
        format_func=lambda x: (
            "Giọng Nam (Nam Minh)" if "Nam" in x else "Giọng Nữ (Hoài Mỹ)"
        ),
    )

    speech_rate = st.select_slider(
        "Tốc độ đọc giọng AI:",
        options=["-10%", "0%", "+10%", "+15%", "+20%", "+30%"],
        value="+10%",
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


# --- HÀM DỊCH TIẾNG VIỆT BẰNG GEMINI API ---
def translate_with_gemini(text_zh, api_key):
    client = genai.Client(api_key=api_key.strip())

    prompt = f"""
    Bạn là một biên tập viên chuyên dịch thuật video Douyin/TikTok ngắn. 
    Hãy dịch toàn bộ đoạn thoại tiếng Trung sau sang Tiếng Việt.
    Yêu cầu:
    - Văn phong tự nhiên, bắt trend, nói chuyện hợp ngữ cảnh video ngắn/review.
    - Không dịch thô/dịch máy, lược bỏ các từ thừa để câu từ ngắn gọn.
    - Chỉ trả về bản dịch Tiếng Việt duy nhất, không giải thích gì thêm.

    Thoại tiếng Trung: "{text_zh}"
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text.strip()


# --- HÀM TẠO VOICE & GHÉP VIDEO ---
async def generate_tts(text, voice, rate, output_path):
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(output_path)


def process_video(video_path, audio_path, output_path, keep_bg, bg_volume):
    video = VideoFileClip(video_path)
    vi_voice = AudioFileClip(audio_path)

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

if st.button("🚀 Bắt Đầu Tự Động Dịch & Lồng Tiếng", type="primary"):
    if not uploaded_video:
        st.error("⚠️ Vui lòng tải video lên trước!")
    elif not gemini_api_key:
        st.error(
            "⚠️ Vui lòng nhập Gemini API Key ở thanh bên trái để AI dịch!"
        )
    else:
        temp_video_path = "temp_input.mp4"
        temp_audio_path = "temp_voice.mp3"
        output_video_path = "output_dubbed.mp4"

        try:
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video.read())

            with st.spinner(
                "1/3 🎧 Whisper AI đang lắng nghe và bóc thoại Tiếng Trung..."
            ):
                zh_text = transcribe_chinese(temp_video_path)
                st.info(f"🗣️ **Thoại tiếng Trung nhận diện được:** {zh_text}")

            with st.spinner(
                "2/3 🤖 Gemini AI đang dịch sang Tiếng Việt mượt mà..."
            ):
                vi_text = translate_with_gemini(zh_text, gemini_api_key)
                st.success(f"📝 **Bản dịch Tiếng Việt (AI):** {vi_text}")

            with st.spinner(
                "3/3 🎬 Đang tạo giọng đọc AI & Ghép vào video..."
            ):
                asyncio.run(
                    generate_tts(
                        vi_text, voice_option, speech_rate, temp_audio_path
                    )
                )
                process_video(
                    temp_video_path,
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
            for p in [temp_video_path, temp_audio_path]:
                if os.path.exists(p):
                    os.remove(p)
