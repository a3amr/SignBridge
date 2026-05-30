import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("WebRTC Test")

from streamlit_webrtc import webrtc_streamer, RTCConfiguration

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302",
                    "stun:stun1.l.google.com:19302",
                ]
            }
        ]
    }
)

webrtc_streamer(
    key="test",
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": True,
        "audio": False,
    }
)

st.write("State:", ctx.state.playing if ctx else "No Context")
