# ruff: noqa: F401
from datetime import timedelta
from pathlib import Path

import pyperclip
import streamlit as st
import torch
import whisper
import yt_dlp

st.set_page_config(
    page_title="Speech to Text",
    page_icon="🔉",
    layout="centered",
    initial_sidebar_state="auto",
)


@st.cache_resource
def load_model():
    return whisper.load_model("base")


def duration_check(info, *, incomplete):
    """Download only videos less than 10 minute (or with unknown duration)"""
    duration = info.get("duration")
    if duration and duration > 601:  # 10 mins limit
        return "The video is too long"


def download_yt_audio(yt_url: str):
    """Download audio from given youtube video URL"""
    # convert cli to python args - https://github.com/yt-dlp/yt-dlp/blob/master/devscripts/cli_to_api.py
    ydl_opts = {
        "match_filter": duration_check,
        "format": "m4a/bestaudio/best",
        "outtmpl": {"default": "audio.%(ext)s"},
        "postprocessors": [
            {  # Extract audio using ffmpeg
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
            }
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download(yt_url)
    print(error_code)


def postprocess_transcription(predictions: dict, include_timestamps: bool):
    if not include_timestamps:
        return predictions.get("text")
    result = []
    for segment in predictions.get("segments", {}):
        startTime = str(0) + str(timedelta(seconds=int(segment["start"]))) + ",000"
        endTime = str(0) + str(timedelta(seconds=int(segment["end"]))) + ",000"
        text = segment["text"]
        segmentId = segment["id"] + 1
        segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] == ' ' else text}\n\n"

        result.append(segment)
    return "".join(result)


def main():
    st.title("Audio|YouTube video Transcription")

    st.sidebar.title("Settings")
    with_timestamps = st.sidebar.selectbox("Include Timestamps", ["Yes", "No"])
    language = st.sidebar.selectbox(
        "Select Language", {"Auto-detect": None, "English": "en"}
    )

    # check GPU
    cuda_available = torch.cuda.is_available()
    print(cuda_available)
    if cuda_available:
        st.info("GPU available 🔥 - Transcriptions will be fast!")
    else:
        st.warning("GPU NOT available 🚨 - Transcriptions might take some time")

    # load model
    model = load_model()
    transcription = None

    # YT video link input
    text_input = st.text_input(label="Enter valid Youtube video URL")

    # audio upload
    audio = st.file_uploader(
        "Upload an audio or short video file",
        type=["mp3", "m4a", "mkv", "mp4"],
    )
    submit_button = st.button(label="Transcribe")

    # submit with video link or uploaded audio
    if submit_button and (text_input or audio):
        st.session_state.text = None
        toast_msg = st.toast("Model is running!", icon="🏃")
        # download audio from YT video url
        if text_input:
            try:
                download_yt_audio(text_input)
            except Exception:
                pass
        elif audio is not None:
            # save uploaded audio
            bytes_data = audio.getvalue()
            with open("./audio.m4a", "wb") as f:
                f.write(bytes_data)

        audio_path = Path("./audio.m4a")
        if not audio_path.exists():
            st.error(
                """Audio file generation failed! Please recheck YouTube URL or uploaded file.
                YT videos only upto 10 mins are supported""",
                icon="🚨",
            )
        else:
            # start transcription
            with st.spinner("Transcribing..."):
                result = model.transcribe(
                    str(audio_path),
                    verbose=True,
                    word_timestamps=True,
                    language=language,
                )
            transcription = postprocess_transcription(result, with_timestamps == "Yes")
            if transcription or st.session_state.text:
                st.session_state.text = transcription
                with st.expander("See Transcription"):
                    st.write(st.session_state.text)
                    # st.code(transcription)
            audio_path.unlink()
    else:
        st.info("Please add YouTube URL or upload audio for transcription", icon="ℹ️")

    # download and copy transcription
    # col1, col2 = st.columns([1, 1])
    # with col1:
    #     copy_btn = st.button("Copy", on_click=update_text, args=[st.session_state.text])
    #     if copy_btn:
    #         pyperclip.copy(transcription)
    #         st.success("Text copied successfully!")
    # with col2:
    #     if not transcription:
    #         transcription = ""
    #     dl_btn = st.download_button(
    #         "Download",
    #         transcription,
    #         "text/plain",
    #     )


if __name__ == "__main__":
    load_model()
    main()
