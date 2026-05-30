# SignBridge
Real-time Sign Language Recognition System leveraging MediaPipe Holistic, LSTM, and SVM. Features live webcam inference, sequence-based landmark tracking, and Text-to-Speech (TTS) integration.

## 📌 Overview
This project is an advanced, real-time Sign Language Recognition System. It utilizes computer vision and deep learning to translate sign language gestures into text and spoken words. The system extracts spatial landmarks from the human body and hands using **MediaPipe Holistic**, processes the sequential data, and classifies the gestures using a robust **Long Short-Term Memory (LSTM)** neural network and **Support Vector Machine (SVM)** models.

## ✨ Features
* **Real-Time Landmark Extraction:** Captures 162 data points (x, y, z coordinates) per frame from the pose, left hand, and right hand using MediaPipe.
* **Sequence-Based Recognition:** Analyzes 30-frame sequences to understand the temporal flow of gestures, rather than static images.
* **Dual Model Approach:** Incorporates both machine learning (SVM) and deep learning (LSTM) architectures for classification and performance comparison.
* **Live Webcam Inference:** Real-time gesture tracking, prediction, and visual feedback directly from a standard webcam.
* **Text-to-Speech (TTS):** Audibly translates recognized sign language gestures into spoken words using `pyttsx3`.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Computer Vision:** OpenCV, MediaPipe
* **Deep Learning:** TensorFlow / Keras (LSTM architecture)
* **Machine Learning:** Scikit-Learn (SVM)
* **Data Manipulation:** NumPy, Pandas
* **Audio Processing:** pyttsx3 (Text-to-Speech)
* **Visualization:** Matplotlib, Seaborn, TensorBoard

## 📂 Project Structure
Currently, the core pipeline is housed within `Sign_Language_system.ipynb`, which handles:
1. Data Collection via webcam.
2. Keypoint extraction and NumPy `.npy` formatting.
3. Neural Network and SVM training.
4. Model evaluation (Confusion Matrices & Accuracy tracking).
5. Real-time implementation.

**Key Assets:**
* `lstm_model.h5`: The trained LSTM neural network model.
* `svm_model.pkl`: The trained SVM classification model.
* `data/`: Categorized datasets stored as Numpy arrays (Sequences of 30 frames per action).
* `Logs/`: TensorBoard logs for monitoring LSTM training metrics.

## 🧠 How It Works
1. **Data Collection:** The system records 30 sequences of 30 frames for each specific sign/word.
2. **Feature Extraction:** MediaPipe isolates the holistic landmarks, flattening them into a dense vector (e.g., 162 spatial coordinates).
3. **Training (LSTM):** The sequential data is fed into a multi-layer LSTM network (128 -> 64 -> 64 units) equipped with Batch Normalization and Dropout layers to learn temporal dependencies.
4. **Inference:** A sliding window of the last 30 frames is continuously fed into the model. If the prediction confidence exceeds the `0.9` threshold, the word is triggered and appended to the current sentence.

## 🚀 Future Roadmap & Improvements
* **Dataset Expansion:** Collecting more sequences per action and applying data augmentation to improve model generalization.
* **Advanced Normalization:** Shifting from raw coordinates to hand-centered relative coordinates for better spatial consistency.
* **Continuous Sign Recognition:** Moving from isolated word classification to continuous sentence modeling using Attention mechanisms or Transformers.
* **Application Deployment:** Refactoring the Jupyter Notebook into a structured modular repository (`src/`, `models/`, `utils/`) and deploying it as a Web App (Streamlit/Flask) or a standalone Desktop Application.
* **Arabic Sign Language (ArSL):** Adding support for Arabic gestures and localized TTS.
