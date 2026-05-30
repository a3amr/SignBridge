import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
import threading

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="SignBridge Demo",
    page_icon="🤟",
    layout="wide"
)

st.title("🤟 SignBridge: Real-Time Sign Language Translation")
st.write("Live sign language recognition using SVM + LSTM")

# =========================
# WEBRTC CONFIG
# =========================

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)

# =========================
# LABELS
# =========================

actions = np.array([
    'about', 'aims', 'are', 'being',
    'developing', 'everyone', 'excited',
    'for', 'graduation', 'hello',
    'here', 'is', 'make', 'NO_Act',
    'our', 'present', 'project',
    'sign language', 'system',
    'thank', 'to', 'Today',
    'we', 'which', 'you'
])

# =========================
# LOAD MODELS
# =========================

@st.cache_resource
def load_models():
    svm = joblib.load("svm_model.pkl")
    lstm = tf.keras.models.load_model("lstm_model.h5")
    return svm, lstm

svm_model, lstm_model = load_models()

# TensorFlow thread safety
predict_lock = threading.Lock()

# =========================
# MEDIAPIPE
# =========================

mp_holistic = mp.solutions.holistic

def extract_keypoints(results):

    pose = (
        np.array([
            [res.x, res.y, res.z]
            for res in results.pose_landmarks.landmark[11:23]
        ])
        if results.pose_landmarks
        else np.zeros((12, 3))
    )

    left_hand = (
        np.array([
            [res.x, res.y, res.z]
            for res in results.left_hand_landmarks.landmark
        ])
        if results.left_hand_landmarks
        else np.zeros((21, 3))
    )

    right_hand = (
        np.array([
            [res.x, res.y, res.z]
            for res in results.right_hand_landmarks.landmark
        ])
        if results.right_hand_landmarks
        else np.zeros((21, 3))
    )

    return np.concatenate([
        pose.flatten(),
        left_hand.flatten(),
        right_hand.flatten()
    ])

# =========================
# VIDEO PROCESSOR
# =========================

class VideoProcessor:

    def __init__(self):

        self.sequence = []
        self.sentence = []

        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def __del__(self):
        try:
            self.holistic.close()
        except:
            pass

    def recv(self, frame):

        try:

            img = frame.to_ndarray(format="bgr24")

            rgb = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            results = self.holistic.process(rgb)

            keypoints = extract_keypoints(results)

            self.sequence.append(keypoints)

            self.sequence = self.sequence[-30:]

            if len(self.sequence) == 30:

                sequence_array = np.expand_dims(
                    self.sequence,
                    axis=0
                )

                flat_sequence = (
                    np.array(self.sequence)
                    .flatten()
                    .reshape(1, -1)
                )

                with predict_lock:

                    lstm_prediction = (
                        lstm_model.predict(
                            sequence_array,
                            verbose=0
                        )[0]
                    )

                svm_prediction = (
                    svm_model.predict_proba(
                        flat_sequence
                    )[0]
                )

                combined = (
                    0.4 * lstm_prediction
                    +
                    0.6 * svm_prediction
                )

                prediction = np.argmax(combined)

                if combined[prediction] > 0.60:

                    action = actions[prediction]

                    if (
                        action != "NO_Act"
                        and
                        (
                            len(self.sentence) == 0
                            or
                            action != self.sentence[-1]
                        )
                    ):
                        self.sentence.append(action)

            display_text = " ".join(
                self.sentence[-3:]
            )

            cv2.putText(
                img,
                display_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            return av.VideoFrame.from_ndarray(
                img,
                format="bgr24"
            )

        except Exception as e:

            print("Frame Processing Error:", e)

            return frame

# =========================
# START WEBRTC
# =========================

webrtc_streamer(
    key="signbridge-demo",

    rtc_configuration=RTC_CONFIGURATION,

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    video_processor_factory=VideoProcessor,

    async_processing=True
)
