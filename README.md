## 🔍 Human Face Emotion Recognition — Gradio Inference App

This repository provides a **production-ready human face emotion recognition system** that performs facial emotion classification from images using a **pre-trained Convolutional Neural Network (CNN)**.

The project is designed **for inference only** and is deployed using **Gradio**, enabling an interactive, web-based interface for real-time predictions.

The application allows users to upload a facial image and receive a predicted emotional state along with class probabilities, while **explicitly enforcing user consent and privacy awareness** prior to any image processing.

---

## 🎯 Key Features

- Pre-trained **CNN model** for human face emotion classification  
- **Inference-only pipeline** (no training or dataset required)  
- **Gradio web interface** for easy image upload and prediction  
- Explicit **privacy & consent mechanism** before processing images  
- Optional **UI anonymization (blurred preview)** for privacy protection  
- Modular, **GitHub-ready code structure** for reproducibility  

---

## 😊 Supported Emotion Classes

The model predicts the following facial emotions:

- Angry  
- Fear  
- Happy  
- Sad  
- Surprise  

> ⚠️ **Note:** In the implementation, the label **"Suprise"** may be preserved to match the original dataset folder naming and training configuration.

---

## 🧠 Model Overview

- **Architecture:** Convolutional Neural Network (CNN)  
- **Input:** RGB facial image resized to **128 × 128**  
- **Output:** Probability distribution over emotion classes  
- **Framework:** TensorFlow / Keras  

---

## 🔗 Pre-trained Model & Notebook Usage

The pre-trained CNN model is **not stored directly in this repository** due to file size constraints.

➡️ **Model Download:** *(https://drive.google.com/file/d/1kB9tL2CMl2dRQViWlZtYZ9fUZJybV2AB/view?usp=sharing)*

The downloaded model is intended to be used with the following notebook:

```text
CV_HFEs_app_gradio.ipynb
