import streamlit as st
from streamlit_webrtc import webrtc_streamer
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

# ⚠️ **تنبيه هام:** ضع هنا أسماء المجلدات (الكلمات) بنفس الترتيب اللي تدرب عليه الموديل
actions = np.array(['about' 'aims' 'are' 'being' 'developing' 'everyone' 'excited' 'for'
 'graduation' 'hello' 'here' 'is' 'make' 'NO_Act' 'our' 'present'
 'project' 'sign language' 'system' 'thank' 'to' 'Today' 'we' 'which'
 'you']) # عدلها حسب الداتا تبعك

# --- تحميل الموديلات (كاش لتسريع الويب) ---
@st.cache_resource
def load_models():
    # استخدمنا اسم الملف اللي حفظته في الكود (svm_model.pkl)
    svm = joblib.load('svm_model.pkl')
    lstm = tf.keras.models.load_model('lstm_model.h5')
    return svm, lstm

svm_model, lstm_model = load_models()

# --- إعداد MediaPipe ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

POSE_WANT = [
    (0, 2), (2, 4), (1, 3), (3, 5), (0, 1), (5, 7), (7, 9), (9, 11),
    (5, 9), (5, 11), (4, 6), (4, 8), (4, 10), (6, 8), (10, 8),
]

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark[11:23]]) if results.pose_landmarks else np.zeros((12, 3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))

    hand_connections = mp_holistic.HAND_CONNECTIONS

    pose_relationships = []
    if POSE_WANT:
        for connection in POSE_WANT:
            pose_relationships.append([pose[connection[0]], pose[connection[1]]])
    
    lh_relationships = []
    if hand_connections:
        for connection in hand_connections:
            lh_relationships.append([lh[connection[0]], lh[connection[1]]])
    
    rh_relationships = []
    if hand_connections:
        for connection in hand_connections:
            rh_relationships.append([rh[connection[0]], rh[connection[1]]])

    all_points = np.concatenate([pose, lh, rh], axis=0)
    all_data = []

    for point in all_points:
        all_data.append(point)

    for relationship in pose_relationships + lh_relationships + rh_relationships:
        all_data.append(relationship[0])
        all_data.append(relationship[1])

    return np.array(all_data)

# --- معالجة الفيديو في الويب ---
class VideoProcessor:
    def __init__(self):
        self.sequence = []
        self.sentence = []
        self.threshold = 0.50
        # model_complexity=0 للسرعة كما طلبت
        self.holistic = mp_holistic.Holistic( min_detection_confidence=0.4, min_tracking_confidence=0.4)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # معالجة الصورة
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = self.holistic.process(img_rgb)
        
        # استخراج النقاط
        keypoints = extract_keypoints(results)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:] # نافذة منزلقة 30 فريم
        
        # التنبؤ
        if len(self.sequence) == 30:
            # 1. LSTM Predict
            res_lstm = lstm_model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
            
            # 2. SVM Predict
            sequence_flat = np.array(self.sequence).reshape(1, -1)
            res_svm_prob = svm_model.predict_proba(sequence_flat)[0]
            
            # 3. دمج النتائج (Ensemble)
            combined_probs = (0.4 * res_lstm) + (0.6 * res_svm_prob)
            combined_prediction = np.argmax(combined_probs)
            conf = combined_probs[combined_prediction]
            
            # Turbo Logic
            if conf > self.threshold:
                action = actions[combined_prediction]
                if action != "NO_Act":
                    if len(self.sentence) == 0 or action != self.sentence[-1]:
                        self.sentence.append(action)
                        self.sequence = self.sequence[-15:] # تقصير الـ Buffer
                        
        if len(self.sentence) > 5:
            self.sentence = self.sentence[-5:]
            
        # رسم الواجهة على الفيديو
        cv2.rectangle(img, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(img, 'Ready: ' + ' '.join(self.sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# تشغيل الكاميرا
webrtc_streamer(key="sign-language-demo", video_processor_factory=VideoProcessor)
