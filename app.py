import asyncio
import json
import os
import re
import edge_tts
from google import genai
from moviepy.editor import AudioFileClip, CompositeAudioClip, VideoFileClip
import moviepy.audio.fx.all as afx
import streamlit as st
import whisper

st.set_page_config(
    page_title="Auto Gensub & Dubbing AI", page_icon="🎬", layout="wide"
)

st.title("🎬 Tool Dịch & Lồng Tiếng Khớp Nhịp Băng Hình 100%")
st.caption("Tự động co giãn tốc độ audio (Time-Stretch) để khớp từng frame hình")

# --- CẤU HÌNH SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Cấu Hình AI")

    gemini_api_key = st.text_input(
        "Nhập Gemini API Key (Miễn phí):", type="password"
    )

    st.subheader("🔊 Tùy Chỉnh Giọng Đọc AI")
    voice_options = {
        "vi-VN-HoaiMyNeural": "Giọng Nữ Review (Hoài Mỹ - Mặc định)",
        "vi-VN-NamMinhNeural": "Giọng Nam Review (Nam Minh)",
    }

    selected_voice = st.selectbox(
        "Chọn giọng đọc AI:",
        options=list(voice_options.keys()),
        format_func=lambda x: voice_options[x],
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


# --- HÀM DỊCH CHUẨN ĐỘ DÀI KHỚP NHỊP ---
def translate_all_segments_batch(segments, api_key):
    if not segments:
        return []

    client = genai.Client(api_key=api_key.strip())

    lines = []
    for i, seg in enumerate(segments):
        duration = seg["end"] - seg["start"]
        text = seg["text"].strip()
        if text:
            lines.append(
                f"{i} (Thời lượng {duration:.1f}s): {text}"
            )

    input_text = "\n".join(lines)

    prompt = f"""
    Bạn là biên dịch viên lồng tiếng chuyên nghiệp cho video ngắn Douyin/TikTok chỉnh ảnh/bóp dáng.
    Dưới đây là danh sách các câu thoại tiếng Trung kèm thời lượng tối đa cho mỗi câu.

    Nhiệm vụ: Dịch sang Tiếng Việt CỰC KỲ NGẮN GỌN sao cho đọc vừa đủ trong thời lượng cho phép.

    QUY TẮC DỊCH CHẮC CHẮN KHỚP NHỊP:
    1. Dùng từ ngữ ngắn gọn, văn phong nói tự nhiên, bắt trend:
       - "P图" -> "chỉnh hình" / "bóp dáng"
       - "背景弯了" -> "hậu cảnh bị méo"
       - "瘦身" / "拉长腿" -> "thon người" / "kéo chân"
    2. Nếu thời lượng câu dưới 2 giây, chỉ dịch từ 3 - 5 từ.
    3. Trả về đúng JSON mảng thuần túy:
    [
      {{"id": 0, "vi": "Nội dung dịch câu 0"}},
      {{"id": 1, "vi": "Nội dung dịch câu 1"}}
    ]

    Danh sách câu thoại:
    {input_text}
    """

    for model_name in ["gemini-3.6-flash", "gemini-2.5-flash"]:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            if response and response.text:
                res_text = response.text.strip()

                if "```json" in res_text:
                    res_text = res_text.split("```json")[1].split("```")[0]
                elif "```" in res_text:
                    res_text = res_text.split("```")[1].split("```")[0]

                parsed_data = json.loads(res_text.strip())
                trans_map = {
                    item["id"]: item.get("vi", "") for item in parsed_data
                }

                results = []
                for i, seg in enumerate(segments):
                    vi_str = trans_map.get(i, "").strip()
                    vi_str = re.sub(r'^["\']|["\']$', "", vi_str)
                    results.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "zh": seg["text"],
                        "vi": vi_str,
                    })
                return results
        except Exception:
            continue

    return [{
        "start": s["start"],
        "end": s["end"],
        "zh": s["text"],
        "vi": "Dịch thất bại",
    } for s in segments]


# --- TẠO FILE ÂM THANH EDGE-TTS ---
async def generate_single_tts_async(text, voice, output_path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def generate_tts_safe(text, voice, output_path):
    clean_text = text.strip()
    if not clean_text:
        return False
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            generate_single_tts_async(clean_text, voice, output_path)
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

if st.button("🚀 Bắt Đầu Ép Khớp Nhịp 100%", type="primary"):
    if not uploaded_video:
        st.error("⚠️ Vui lòng tải video lên trước!")
    elif not gemini_api_key:
        st.error("⚠️ Vui lòng nhập Gemini API Key!")
    else:
        output_video_path = "output_dubbed.mp4"
        temp_files = []

        try:
            # 1. Bóc thoại bằng Whisper
            with st.spinner("1/3 🎧 Phân tích chính xác từng milisecond..."):
                segments = transcribe_chinese_segments("temp_input.mp4")

            # 2. Dịch bằng Gemini
            with st.spinner("2/3 🤖 Dịch chuẩn độ dài kịch bản..."):
                translated_segments = translate_all_segments_batch(
                    segments, gemini_api_key
                )

            # 3. Lồng tiếng & Co giãn tốc độ (Time-Stretch)
            with st.spinner("3/3 🎬 Đang ép tốc độ âm thanh chuẩn hình..."):
                audio_clips = []

                for i, item in enumerate(translated_segments):
                    if not item["vi"].strip():
                        continue

                    audio_filename = f"temp_seg_{i}.mp3"
                    success = generate_tts_safe(
                        item["vi"], selected_voice, audio_filename
                    )

                    if success:
                        temp_files.append(audio_filename)
                        clip = AudioFileClip(audio_filename)

                        target_duration = item["end"] - item["start"]
                        if target_duration > 0.3:
                            # Tính tỷ lệ speedup/slowdown để vừa khít mốc thời gian gốc
                            factor = clip.duration / target_duration
                            # Giới hạn tỷ lệ co giãn từ 0.7x - 1.5x để giọng không bị méo
                            factor = max(0.7, min(1.5, factor))
                            clip = clip.fx(afx.time_stretch, factor)

                        clip = clip.set_start(item["start"])
                        audio_clips.append(clip)

                video = VideoFileClip("temp_input.mp4")

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
                    label="📥 Tải Video Đã Khớp Nhịp",
                    data=file,
                    file_name="douyin_dubbed_synced.mp4",
                    mime="video/mp4",
                )

        except Exception as e:
            st.error(f"❌ Xảy ra lỗi: {str(e)}")

        finally:
            if os.path.exists("temp_input.mp4"):
                os.remove("temp_input.mp4")
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)
