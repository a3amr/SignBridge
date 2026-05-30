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
st.markdown("قم بتشغيل الكاميرا، وانتظر ثواني حتى يتم تحميل النماذج، ثم ابدأ الإشارة!")

# --- إعدادات الاتصال (مهمة جداً لحل خطأ الـ ICE) ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- الكلمات (تأكد أن القائمة هنا مطابقة تماماً لتدريبك) ---
actions = np.array(['about', 'aims', 'are', 'being', 'developing', 'everyone', 'excited', 'for',
 'graduation', 'hello', 'here', 'is', 'make', 'NO_Act', 'our', 'present',
 'project', 'sign language', 'system', 'thank', 'to', 'Today', 'we', 'which',
 'you']) 

# --- تحميل الموديلات ---
@st.cache_resource
def load_models():
    svm = joblib.load('svm_model.pkl')
    lstm = tf.keras.models.load_model('lstm_model.h5')
    return svm, lstm

try:
    svm_model, lstm_model = load_models()
except Exception as e:
    st.error(f"خطأ في تحميل الموديلات: {e}")

# --- إعداد MediaPipe ---
mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark[11:23]]) if results.pose_landmarks else np.zeros((12, 3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    
    # دمج البيانات المسطحة فقط (لتجنب تعقيد الأبعاد)
    return np.concatenate([pose.flatten(), lh.flatten(), rh.flatten()])

# --- معالجة الفيديو ---
class VideoProcessor:
    def __init__(self):
        self.sequence = []
        self.sentence = []
        self.threshold = 0.50
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)
        
        keypoints = extract_keypoints(results)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:] 
        
        if len(self.sequence) == 30:
            seq_array = np.expand_dims(self.sequence, axis=0)
            
            # 1. LSTM Predict
            res_lstm = lstm_model.predict(seq_array, verbose=0)[0]
            
            # 2. SVM Predict (تأكد من شكل المدخلات)
            sequence_flat = np.array(self.sequence).flatten().reshape(1, -1)
            res_svm_prob = svm_model.predict_proba(sequence_flat)[0]
            
            combined_probs = (0.4 * res_lstm) + (0.6 * res_svm_prob)
            combined_prediction = np.argmax(combined_probs)
            conf = combined_probs[combined_prediction]
            
            if conf > self.threshold:
                action = actions[combined_prediction]
                if action != "NO_Act":
                    if len(self.sentence) == 0 or action != self.sentence[-1]:
                        self.sentence.append(action)
                        
        if len(self.sentence) > 5:
            self.sentence = self.sentence[-5:]
            
        cv2.rectangle(img, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(img, ' '.join(self.sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# تشغيل الكاميرا مع إضافة الإعدادات
webrtc_streamer(
    key="sign-language-demo", 
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION
)
