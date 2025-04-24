# Real-Time Facial Recognition System

A real-time facial recognition application developed using deep learning and computer vision techniques. This project was driven by my passion for AI and a desire to build a complete ML pipeline – from dataset collection and model training to deployment in a real-time GUI application.

## 🚀 Features
- Real-time face detection and recognition
- One-shot learning using Siamese Neural Networks
- User-friendly GUI built with Kivy
- Webcam integration for live predictions
- High accuracy on test data

## 📚 Technologies Used
- **TensorFlow Keras** – Model development and training
- **OpenCV** – Face detection and webcam integration
- **Kivy** – GUI for real-time face recognition
- **NumPy & Pandas** – Data preprocessing and handling
- **Labeled Faces in the Wild (LFW)** – Dataset for training
- **Siamese Neural Networks** – One-shot image recognition architecture

## 🧠 Model Architecture
Inspired by the **"Siamese Neural Networks for One-shot Image Recognition"** paper, the model compares two face images and predicts whether they belong to the same person. This enables high-accuracy recognition with minimal training data.

## 🗂️ Dataset
- **LFW Dataset**: Pre-labeled celebrity face images
- **Custom Samples**: Captured via webcam for personalized training
- **Data Augmentation**: Applied transformations to improve model robustness

## 🔧 How It Works
1. Preprocess face images using OpenCV
2. Train a Siamese Network model on the LFW dataset and augmented webcam samples
3. Use the trained model in a Kivy-based GUI to detect and recognize faces in real-time

## 🎯 Accuracy
Achieved **100% accuracy on test data** during evaluation – thanks to effective preprocessing, data augmentation, and model tuning.

## 🧪 Skills Gained
- Deep learning with TensorFlow and Keras
- Real-time application development
- Face detection with OpenCV
- GUI design using Kivy
- Debugging and model tuning techniques

