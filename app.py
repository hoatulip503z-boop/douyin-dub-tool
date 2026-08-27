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
    "Tự động bóc thoại -> Dịch gộp 100% Tiếng Việt -> Lồng tiếng chuẩn mốc"
    " thời gian"
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


# --- HÀM DỊCH GỘP TẤT CẢ CÂU SANG TIẾNG VIỆT (TRÁNH RATE LIMIT) ---
def translate_all_segments_batch(segments, api_key):
    if not segments:
        return []

    client = genai.Client(api_key=api_key.strip())

    # Tạo danh sách câu thoại dạng index: câu tiếng trung
    lines = [
        f"{i}: {seg['text'].strip()}"
        for i, seg in enumerate(segments)
        if seg["text"].strip()
    ]
    input_text = "\n".join(lines)

    prompt = f"""
    Bạn là một biên tập viên vietsub và lồng tiếng video ngắn Douyin/TikTok (Mẹo chỉnh ảnh, Review).
    Dưới đây là danh sách các câu thoại tiếng Trung theo số thứ tự dòng.

    NHIỆM VỤ: Dịch TẤT CẢ các câu thoại này sang TIẾNG VIỆT.
    YÊU CẦU BẮT BUỘC:
    1. Văn phong: CỰC KỲ NGẮN GỌN, TỰ NHIÊN, BẮT TREND (để vừa khớp với thời gian đọc video ngắn).
    2. Tuyệt đối CHỈ DỊCH SANG TIẾNG VIỆT, KHÔNG giữ lại tiếng Trung.
    3. Định dạng đầu ra bắt buộc là một mảng JSON thuần túy theo mẫu:
    [
      {{"id": 0, "vi": "Nội dung dịch câu 0"}},
      {{"id": 1, "vi": "Nội dung dịch câu 1"}}
    ]
    Không viết thêm bất kỳ câu giải thích hay văn bản nào khác ngoài đoạn JSON.

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

                # Làm sạch chuỗi JSON nếu có thẻ markdown
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

    # Nếu lỗi JSON, trả về kết quả dự phòng
    return [{
        "start": s["start"],
        "end": s["end"],
        "zh": s["text"],
        "vi": "Dịch thất bại, vui lòng kiểm tra lại API Key",
    } for s in segments]


# --- TẠO FILE ÂM THANH AN TOÀN ---
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
                "1/3 🎧 Whisper AI đang phân tích và nhận diện mốc thời"
                " gian..."
            ):
                segments = transcribe_chinese_segments("temp_input.mp4")

            if not segments:
                st.warning("Không tìm thấy lời thoại trong video!")

            # 2. Dịch toàn bộ sang Tiếng Việt trong 1 lần gọi (Batch)
            with st.spinner(
                "2/3 🤖 Gemini AI đang dịch toàn bộ câu thoại sang Tiếng Việt"
                " mượt mà..."
            ):
                translated_segments = translate_all_segments_batch(
                    segments, gemini_api_key
                )

            # Hiển thị kết quả dịch Tiếng Việt chuẩn
            st.success("📝 **Bản dịch Tiếng Việt từng mốc thời gian:**")
            for item in translated_segments:
                if item["vi"].strip():
                    st.write(
                        f"⏱️ **[{item['start']:.1f}s - {item['end']:.1f}s]** :"
                        f" {item['vi']}"
                    )

            # 3. Lồng tiếng và ghép khớp thời gian
            with st.spinner(
                "3/3 🎬 Đang tạo giọng đọc AI & Ghép đúng thời điểm vào"
                " video..."
            ):
                audio_clips = []

                for i, item in enumerate(translated_segments):
                    if not item["vi"].strip():
                        continue

                    audio_filename = f"temp_seg_{i}.mp3"

                    # Tạo file âm thanh từng câu
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
