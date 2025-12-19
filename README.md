# 🔍 Human Face Emotion Recognition — Gradio Inference App

This repository provides a **production-ready emotion recognition system** that performs **facial emotion classification from images** using a **pre-trained Convolutional Neural Network (CNN)**.  
The project is designed for **inference only** and is deployed using **Gradio** for interactive, web-based predictions.

The application allows users to upload a facial image and receive a **predicted emotional state** along with **class probabilities**, while explicitly enforcing **user consent and privacy awareness** prior to image processing.

---

## 🎯 Key Features

- Pre-trained CNN model for human face emotion classification  
- Prediction-only pipeline (no training or dataset required)  
- Gradio web interface for easy image upload and inference  
- Explicit privacy & consent mechanism before processing images  
- Modular, GitHub-ready code structure for reproducibility  
- Hugging Face Spaces compatible for public or private deployment  

---

## 😊 Supported Emotion Classes

The model predicts the following facial emotions:

- Angry  
- Fear  
- Happy  
- Sad  
- Surprise  

---

## 🧠 Model Overview

- **Architecture:** Convolutional Neural Network (CNN)  
- **Input:** RGB facial image resized to **128 × 128**  
- **Output:** Probability distribution over emotion classes  
- **Framework:** TensorFlow / Keras  

The trained model is stored as:

```text
models/hfe_emotion_cnn.keras
