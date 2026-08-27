import asyncio
import json
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
    "Tự động bóc thoại -> Dịch chuẩn Review -> Tự động căn chỉnh tốc độ tránh"
    " đọc đè"
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
        "Tốc độ đọc giọng AI mặc định:",
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


# --- HÀM DỊCH GỘP BẰNG GEMINI CHUẨN THUẬT NGỮ REVIEW ---
def translate_all_segments_batch(segments, api_key):
    if not segments:
        return []

    client = genai.Client(api_key=api_key.strip())

    lines = [
        f"{i}: {seg['text'].strip()}"
        for i, seg in enumerate(segments)
        if seg["text"].strip()
    ]
    input_text = "\n".join(lines)

    prompt = f"""
    Bạn là một chuyên gia vietsub và lồng tiếng cho các video ngắn Douyin/TikTok về hướng dẫn chỉnh ảnh, bóp dáng, làm đẹp (Review app chỉnh ảnh/video).
    Dưới đây là danh sách các câu thoại tiếng Trung theo số thứ tự.

    Nhiệm vụ: Dịch TẤT CẢ các câu thoại này sang TIẾNG VIỆT tự nhiên nhất.

    QUY TẮC CHUYỂN NGỮ BẮT BUỘC (QUAN TRỌNG):
    1. Các từ lóng/thuật ngữ chỉnh ảnh tiếng Trung phải dịch tự nhiên sang Tiếng Việt:
       - "P图" / "修图" -> Dịch là "chỉnh ảnh", "bóp dáng", "sửa hình" (TUYỆT ĐỐI KHÔNG DỊCH THÀNH chữ "P" đơn lẻ).
       - "拉长腿" -> Dịch là "kéo dài chân".
       - "瘦身" / "瘦腿" -> Dịch là "thon gọn người" / "làm thon chân".
       - "背景保护" -> Dịch là "bảo vệ phông nền" / "giữ phông nền".
    2. Câu dịch phải CỰC KỲ NGẮN GỌN (dưới 10 từ/câu) để đảm bảo không bị đọc đè sang câu sau.
    3. Định dạng đầu ra BẮT BUỘC là mảng JSON thuần túy theo mẫu:
    [
      {{"id": 0, "vi": "Nội dung dịch câu 0"}},
      {{"id": 1, "vi": "Nội dung dịch câu 1"}}
    ]
    Không viết thêm bất kỳ câu giải thích nào khác ngoài đoạn JSON.

    Danh sách câu tiếng Trung:
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
        "vi": "Dịch thất bại, hãy thử lại.",
    } for s in segments]


# --- TẠO FILE ÂM THANH EDGE-TTS ---
async def generate_single_tts_async(text, voice, rate, output_path):
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(output_path)


def generate_tts_safe(text, voice, rate, output_path):
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
            # 1. Bóc thoại bằng Whisper
            with st.spinner(
                "1/3 🎧 Whisper AI đang phân tích mốc thời gian từng câu..."
            ):
                segments = transcribe_chinese_segments("temp_input.mp4")

            if not segments:
                st.warning("Không tìm thấy lời thoại trong video!")

            # 2. Dịch bằng Gemini AI
            with st.spinner(
                "2/3 🤖 Gemini AI đang dịch câu ngắn gọn chuẩn"
                " GenSubAI..."
            ):
                translated_segments = translate_all_segments_batch(
                    segments, gemini_api_key
                )

            # Hiển thị bản dịch
            st.success("📝 **Bản dịch Tiếng Việt từng mốc thời gian:**")
            for item in translated_segments:
                if item["vi"].strip():
                    st.write(
                        f"⏱️ **[{item['start']:.1f}s - {item['end']:.1f}s]** :"
                        f" {item['vi']}"
                    )

            # 3. Lồng tiếng & Căn chỉnh chống trùng lặp
            with st.spinner(
                "3/3 🎬 Đang xử lý âm thanh & Căn chỉnh chống chồng tiếng..."
            ):
                audio_clips = []

                for i, item in enumerate(translated_segments):
                    if not item["vi"].strip():
                        continue

                    audio_filename = f"temp_seg_{i}.mp3"
                    success = generate_tts_safe(
                        item["vi"],
                        selected_voice,
                        speech_rate,
                        audio_filename,
                    )

                    if success:
                        temp_files.append(audio_filename)
                        clip = AudioFileClip(audio_filename)

                        # Thời lượng cho phép của câu thoại này trong video
                        allowed_duration = item["end"] - item["start"]

                        # Nếu file âm thanh dài hơn mốc thời gian cho phép -> Tự động ép tốc độ ngắn lại
                        if clip.duration > allowed_duration and allowed_duration > 0.3:
                            speed_factor = clip.duration / allowed_duration
                            clip = clip.speedx(speed_factor)

                        # Giới hạn cứng độ dài clip để tuyệt đối không tràn sang câu tiếp theo
                        if allowed_duration > 0:
                            clip = clip.subclip(0, min(clip.duration, allowed_duration))

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
