import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

st.title("WebRTC T-est")

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    }
)

ctx = webrtc_streamer(
    key="test",
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            "width": 320,
            "height": 240,
        },
        "audio": False,
    },
    async_processing=False,
)

st.write("State:", ctx.state.playing if ctx else None)
