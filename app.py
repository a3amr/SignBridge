import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("Camera Test")

webrtc_streamer(
    key="test",
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)
