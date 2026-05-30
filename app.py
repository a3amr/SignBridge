import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# 1. إعداد واجهة الويب
st.set_page_config(page_title="SignBridge Demo", page_icon="🤟")
st.title("🤟 SignBridge: Real-Time Sign Language Translation")
st.markdown("قم بتشغيل الكاميرا وانتظر قليلاً حتى يبدأ النظام، ثم ابدأ بالإشارة!")

# 2. تحديد الكلمات (يجب أن تكون نفس الكلمات اللي دربت عليها الموديل)
# قم بتعديل هذه المصفوفة بناءً على الداتا تبعتك
actions = np.array(['hello', 'today', 'everyone']) 

# 3. تحميل الموديل مرة واحدة فقط لتسريع النظام
@st.cache_resource
def load_lstm_model():
    return load_model('lstm_model.h5')
    
model = load_lstm_model()

# 4. إعداد MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# دالة استخراج النقاط - **يجب أن تنسخ دالتك الأصلية من ملف الـ Notebook وتضعها هنا**
def extract_keypoints(results):
    # كمثال مبدئي:
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# 5. معالجة الفيديو القادم من الكاميرا
class VideoProcessor:
    def __init__(self):
        self.sequence = []
        self.sentence = []
        self.threshold = 0.9
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # استخراج المعالم
        results = self.holistic.process(img_rgb)
        keypoints = extract_keypoints(results)
        
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:] # الاحتفاظ بآخر 30 فريم فقط
        
        # التنبؤ
        if len(self.sequence) == 30:
            # نتأكد إن الـ shape متطابق مع موديلك (1, 30, num_features)
            res = model.predict(np.expand_dims(self.sequence, axis=0))[0]
            
            if res[np.argmax(res)] > self.threshold:
                word = actions[np.argmax(res)]
                if len(self.sentence) > 0:
                    if self.sentence[-1] != word:
                        self.sentence.append(word)
                else:
                    self.sentence.append(word)
                    
        if len(self.sentence) > 5:
            self.sentence = self.sentence[-5:]
            
        # رسم الجملة على الشاشة
        cv2.rectangle(img, (0,0), (640, 40), (24, 24, 24), -1)
        cv2.putText(img, ' '.join(self.sentence), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 6. تشغيل الكاميرا
webrtc_streamer(key="sign-language-demo", video_processor_factory=VideoProcessor)
