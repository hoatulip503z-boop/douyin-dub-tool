import asyncio
import os
import re
import edge_tts
from google import genai
from moviepy.editor import AudioFileClip, CompositeAudioClip, VideoFileClip
import streamlit as st
import whisper

st.set_page_config(
    page_title="Auto Gensub & Dubbing AI", page_icon="🎬", layout="wide"
)

st.title("🎬 Tool Tự Động Dịch & Lồng Tiếng Khớp Nhịp (GenSubAI Style)")
st.caption(
    "Tự động nhận diện chuẩn timestamp -> Dịch mượt ngắn gọn -> Ghép thoại"
    " đúng khớp mốc thời gian"
)

# --- CẤU HÌNH SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Cấu Hình AI")

    gemini_api_key = st.text_input(
        "Nhập Gemini API Key (Miễn phí):", type="password"
    )

    st.subheader("🔊 Tùy Chỉnh Giọng Đọc AI")
    voice_options = {
        "vi-VN-HoaiMyNeural": "Giọng Nữ Nữ tính / Review (Hoài Mỹ)",
        "vi-VN-NamMinhNeural": "Giọng Nam Tự nhiên / Đọc nhanh (Nam Minh)",
        "vi-VN-NamMinhMultilingualNeural": (
            "Giọng Nam Truyền cảm / Chuẩn (Nam Minh Pro)"
        ),
    }

    selected_voice = st.selectbox(
        "Chọn giọng đọc AI:",
        options=list(voice_options.keys()),
        format_func=lambda x: voice_options[x],
    )

    speech_rate = st.select_slider(
        "Tốc độ đọc giọng AI:",
        options=["-10%", "+0%", "+5%", "+10%", "+15%"],
        value="+0%",
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


def transcribe_chinese_segments(video_path):
    model = load_whisper_model()
    result = model.transcribe(video_path, language="zh")
    return result.get("segments", [])


# --- HÀM DỊCH TIẾNG VIỆT TỪNG ĐOẠN BẰNG GEMINI ---
def translate_segment_with_gemini(text_zh, api_key):
    clean_text = text_zh.strip()
    if not clean_text:
        return ""

    client = genai.Client(api_key=api_key.strip())

    prompt = f"""
    Bạn là biên tập viên vietsub/lồng tiếng cho các video Douyin/TikTok ngắn (Chỉnh ảnh, Review, Mẹo vặt).
    Hãy dịch duy nhất câu thoại tiếng Trung sau sang Tiếng Việt.

    YÊU CẦU BẮT BUỘC:
    1. Văn phong: Cực kỳ BẮT TREND, TỰ NHIÊN và NGẮN GỌN (khớp thời gian thoại).
    2. Không dịch rườm rà, giải thích. Chỉ xuất ra duy nhất 1 câu Tiếng Việt ngắn gọn.

    Tiếng Trung: "{clean_text}"
    """

    for model_name in ["gemini-3.6-flash", "gemini-2.5-flash"]:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            if response and response.text:
                translated = response.text.strip()
                # Lược bỏ ký tự thừa hoặc ngoặc kép nếu có
                translated = re.sub(r'^["\']|["\']$', "", translated)
                return translated
        except Exception:
            continue
    return clean_text


# --- XỬ LÝ TẠO FILE ÂM THANH AN TOÀN ---
async def generate_single_tts_async(text, voice, rate, output_path):
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(output_path)


def generate_tts_safe(text, voice, rate, output_path):
    # Loại bỏ khoảng trắng và ký tự không hợp lệ
    clean_text = text.strip()
    if not clean_text:
        return False
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            generate_single_tts_async(clean_text, voice, rate, output_path)
        )
        loop.close()
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception:
        return False


# --- GIAO DIỆN CHÍNH ---
uploaded_video = st.file_uploader(
    "Tải video Douyin gốc tiếng Trung lên (.mp4)", type=["mp4", "mov"]
)

if uploaded_video:
    temp_video_path = "temp_input.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.read())

if st.button("🚀 Bắt Đầu Tự Động Dịch & Lồng Tiếng Khớp Nhịp", type="primary"):
    if not uploaded_video:
        st.error("⚠️ Vui lòng tải video lên trước!")
    elif not gemini_api_key:
        st.error(
            "⚠️ Vui lòng nhập Gemini API Key ở thanh bên trái để AI dịch!"
        )
    else:
        output_video_path = "output_dubbed.mp4"
        temp_files = []

        try:
            # 1. Bóc thoại theo mốc thời gian
            with st.spinner(
                "1/3 🎧 Whisper AI đang phân tích và ghi nhận mốc thời gian"
                " từng câu..."
            ):
                segments = transcribe_chinese_segments("temp_input.mp4")

            if not segments:
                st.warning("Không tìm thấy lời thoại trong video!")

            # 2. Dịch từng segment
            with st.spinner(
                "2/3 🤖 Gemini AI đang dịch từng câu ngắn gọn chuẩn"
                " GenSubAI..."
            ):
                translated_segments = []
                for seg in segments:
                    vi_text = translate_segment_with_gemini(
                        seg["text"], gemini_api_key
                    )
                    if vi_text.strip():
                        translated_segments.append({
                            "start": seg["start"],
                            "end": seg["end"],
                            "vi": vi_text,
                        })

            # Hiển thị kết quả dịch từng câu
            st.success("📝 **Bản dịch từng mốc thời gian:**")
            for item in translated_segments:
                st.write(
                    f"⏱️ **[{item['start']:.1f}s - {item['end']:.1f}s]** :"
                    f" {item['vi']}"
                )

            # 3. Tạo file audio lồng ghép chuẩn timestamp
            with st.spinner(
                "3/3 🎬 Đang tạo giọng đọc AI & Ghép khớp từng giây vào"
                " video..."
            ):
                audio_clips = []

                for i, item in enumerate(translated_segments):
                    audio_filename = f"temp_seg_{i}.mp3"

                    # Tạo file giọng đọc an toàn
                    success = generate_tts_safe(
                        item["vi"],
                        selected_voice,
                        speech_rate,
                        audio_filename,
                    )

                    if success:
                        temp_files.append(audio_filename)
                        clip = AudioFileClip(audio_filename).set_start(
                            item["start"]
                        )
                        audio_clips.append(clip)

                video = VideoFileClip("temp_input.mp4")

                # Trộn các đoạn voice đã lồng tiếng
                if audio_clips:
                    dubbed_audio = CompositeAudioClip(audio_clips)

                    if keep_bg_music and video.audio is not None:
                        bg_audio = video.audio.volumex(bg_vol / 100.0)
                        final_audio = CompositeAudioClip(
                            [bg_audio, dubbed_audio]
                        )
                    else:
                        final_audio = dubbed_audio

                    final_video = video.set_audio(final_audio)
                else:
                    final_video = video

                final_video.write_videofile(
                    output_video_path,
                    codec="libx264",
                    audio_codec="aac",
                    fps=video.fps,
                )

                video.close()
                for c in audio_clips:
                    c.close()

            st.balloons()
            st.video(output_video_path)

            with open(output_video_path, "rb") as file:
                st.download_button(
                    label="📥 Tải Video Đã Lồng Tiếng Về Máy",
                    data=file,
                    file_name="douyin_dubbed_synced.mp4",
                    mime="video/mp4",
                )

        except Exception as e:
            st.error(f"❌ Xảy ra lỗi trong quá trình xử lý: {str(e)}")

        finally:
            if os.path.exists("temp_input.mp4"):
                os.remove("temp_input.mp4")
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)
