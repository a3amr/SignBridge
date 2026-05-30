import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib

# --- إعداد الصفحة ---
st.set_page_config(page_title="SignBridge Demo", page_icon="🤟")
st.title("🤟 SignBridge: Real-Time Sign Language Translation")

# إعدادات اتصال مرنة جداً
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

actions = np.array(['about', 'aims', 'are', 'being', 'developing', 'everyone', 'excited', 'for',
 'graduation', 'hello', 'here', 'is', 'make', 'NO_Act', 'our', 'present',
 'project', 'sign language', 'system', 'thank', 'to', 'Today', 'we', 'which',
 'you']) 

@st.cache_resource
def load_models():
    svm = joblib.load('svm_model.pkl')
    lstm = tf.keras.models.load_model('lstm_model.h5')
    return svm, lstm

svm_model, lstm_model = load_models()
mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark[11:23]]) if results.pose_landmarks else np.zeros((12, 3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([pose.flatten(), lh.flatten(), rh.flatten()])

class VideoProcessor:
    def __init__(self):
        self.sequence = []
        self.sentence = []
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)
        
        keypoints = extract_keypoints(results)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:] 
        
        if len(self.sequence) == 30:
            seq_array = np.expand_dims(self.sequence, axis=0)
            # التنبؤ
            res_lstm = lstm_model.predict(seq_array, verbose=0)[0]
            seq_flat = np.array(self.sequence).flatten().reshape(1, -1)
            res_svm = svm_model.predict_proba(seq_flat)[0]
            
            combined = (0.4 * res_lstm) + (0.6 * res_svm)
            pred = np.argmax(combined)
            
            if combined[pred] > 0.6:
                action = actions[pred]
                if action != "NO_Act" and (len(self.sentence) == 0 or action != self.sentence[-1]):
                    self.sentence.append(action)
        
        # عرض النتائج
        cv2.putText(img, ' '.join(self.sentence[-3:]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# التشغيل مع إعدادات الضغط المخفف
webrtc_streamer(
    key="sign-language-demo",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
