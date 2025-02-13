# Voice Emotion Recognition in Python

## 📌 Project Description
This project is a **Voice Emotion Recognition** system built using **Python**. It processes audio files, extracts features, and classifies emotions such as **happy, sad, angry, neutral**, etc. The model is trained using machine learning and deep learning techniques on emotion datasets like **RAVDESS, CREMA-D, TESS, or EmoDB**.

## 🛠 Features
- Feature extraction using **MFCC, Chroma, and Mel Spectrogram**.
- Uses **TensorFlow/Keras, Librosa, Scikit-learn** for training models.
- Supports both real-time microphone input and pre-recorded audio files.
- Emotion classification with deep learning (CNN, RNN, LSTMs, etc.).
- Visual representation of waveform and spectrogram.

## 📂 Folder Structure
```
📦 Voice-Emotion-Recognition
 ┣ 📂 data           # Dataset files
 ┣ 📂 models         # Pretrained models
 ┣ 📂 notebooks      # Jupyter Notebooks for training and testing
 ┣ 📂 src            # Source code files
 ┃ ┣ 📜 feature_extraction.py  # Extracts features from audio
 ┃ ┣ 📜 train_model.py         # Training script
 ┃ ┣ 📜 predict.py             # Prediction script
 ┣ 📜 requirements.txt  # Dependencies
 ┣ 📜 README.md         # Project documentation
```

## 📦 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/emotion-recognition.git
cd emotion-recognition
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

## 🎤 Usage
### Predict Emotion from an Audio File
```bash
python predict.py --file sample_audio.wav
```
### Train the Model
```bash
python train_model.py --dataset data/
```
### Live Emotion Recognition
```bash
python predict.py --mic
```

## 🏗 Dependencies
- Python 3.8+
- TensorFlow/Keras
- Librosa
- Scikit-learn
- Numpy
- Matplotlib
- Sounddevice
- Pyaudio

## 📊 Dataset Sources
- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **CREMA-D** (Crowd-Sourced Emotional Multimodal Actors Dataset)
- **TESS** (Toronto Emotional Speech Set)
- **EmoDB** (Berlin Database of Emotional Speech)

## 📌 Contributing
Feel free to contribute! You can submit **issues** or **pull requests** to improve the model, add new features, or fix bugs.

## 📜 License
This project is licensed under the **MIT License**.

## 🙌 Acknowledgements
Thanks to the open-source community and researchers for developing datasets and libraries that made this project possible.

---
📧 **Contact:** ytreshlol202@gmail.com | LinkedIn: [Mehdi](https://www.linkedin.com/in/mehdi-dinari-b0487a2a9/)
