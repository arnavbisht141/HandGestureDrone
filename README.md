#Gesture Controlled Drone System
This repository contains an AIMS DTU project focused on controlling a drone using hand gestures in real time. The project combines computer vision and deep learning to study human–drone interaction with an emphasis on reliability, safety, and real-time performance.

#Overview
The system processes a live camera feed to detect hand gestures and map them to drone actions. MediaPipe is used for hand detection and landmark extraction, while a custom convolutional neural network (CNN) is trained to classify directional gestures from cropped hand images.
Drone commands are currently implemented as dummy functions.

#Technologies Used:
Python
OpenCV
MediaPipe (Tasks API)
TensorFlow / Keras

#Gesture Recognition Pipeline:
Webcam input
MediaPipe hand detection and cropping
CNN-based gesture classification
State-based control logic
Drone command execution (dummy)

#Supported Gestures
CNN-based gestures:
Stop
Up
Down
Left
Right
Logic-based system gestures:
Photo capture with countdown
Follow-me mode using face detection
Return-to-user and shutdown sequence
#Dataset
Dataset is self-collected using a webcam
Hand images are cropped using MediaPipe during capture
Dataset split: 80% training, 20% validation
Project Structure
Copy code

dataset/
  train/
  val/
cnn/
  train_cnn_gesture.py
  cnn_live_inference.py
hand_landmarker.task
gesture_control.py
Notes
This project focuses on perception and control logic, not hardware integration.
All drone-related actions are placeholders for testing and demonstration.
Future Work
Extend gesture set
Improve robustness using data augmentation
Temporal gesture recognition
Edge deployment and hardware integration
Author
Arnav Bisht
AIMS DTU
