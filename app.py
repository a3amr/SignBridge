import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("WebRTC Test")

ctx = webrtc_streamer(
    key="test",
    media_stream_constraints={
        "video": True,
        "audio": False,
    }
)

st.write("State:", ctx.state.playing if ctx else "No Context")
