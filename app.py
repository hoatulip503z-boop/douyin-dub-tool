import asyncio
import os
import edge_tts
from moviepy.editor import AudioFileClip, CompositeAudioClip, VideoFileClip
import streamlit as st

st.set_page_config(
    page_title="Auto Dubbing Studio Free", page_icon="🎬", layout="wide"
)

st.title("🎬 Tool Dịch & Lồng Tiếng Video Douyin (100% Free)")
st.caption("Tự động đọc bản dịch tiếng Việt, ghép khớp Timeline và xuất Video")

with st.sidebar:
    st.header("⚙️ Tùy Chỉnh Âm Thanh")
    voice_option = st.selectbox(
        "Chọn giọng đọc AI:",
        options=["vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"],
        format_func=lambda x: (
            "Giọng Nam (Nam Minh)"
            if "Nam" in x
            else "Giọng Nữ (Hoài Mỹ)"
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

col1, col2 = st.columns(2)

with col1:
    uploaded_video = st.file_uploader(
        "1. Tải video Douyin gốc lên (.mp4)", type=["mp4", "mov"]
    )

with col2:
    vietnamese_text = st.text_area(
        "2. Dán bản dịch Tiếng Việt vào đây:",
        height=200,
        placeholder="Nhập toàn bộ kịch bản/thoại tiếng Việt...",
    )


async def generate_tts(text, voice, rate, output_path):
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(output_path)


def process_video(
    video_path,
    audio_path,
    output_path,
    keep_bg,
    bg_volume,
):
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


if st.button("🚀 Bắt đầu Lồng Tiếng & Render Video", type="primary"):
    if not uploaded_video or not vietnamese_text.strip():
        st.error(
            "⚠️ Vui lòng tải video lên và điền bản dịch tiếng Việt trước!"
        )
    else:
        with st.spinner("⏳ Đang tạo giọng đọc AI tiếng Việt và ghép vào video..."):
            temp_video_path = "temp_input.mp4"
            temp_audio_path = "temp_voice.mp3"
            output_video_path = "output_dubbed.mp4"

            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video.read())

            asyncio.run(
                generate_tts(
                    vietnamese_text,
                    voice_option,
                    speech_rate,
                    temp_audio_path,
                )
            )

            process_video(
                temp_video_path,
                temp_audio_path,
                output_video_path,
                keep_bg_music,
                bg_vol,
            )

            st.success("🎉 Xuất video hoàn tất!")

            st.video(output_video_path)

            with open(output_video_path, "rb") as file:
                st.download_button(
                    label="📥 Tải Video Đã Lồng Tiếng Về Máy",
                    data=file,
                    file_name="video_dubbed_vietnamese.mp4",
                    mime="video/mp4",
                )

            for p in [temp_video_path, temp_audio_path]:
                if os.path.exists(p):
                    os.remove(p)