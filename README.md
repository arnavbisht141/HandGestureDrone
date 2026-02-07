# Gesture Controlled Drone System

This repository contains an AIMS DTU project focused on controlling a drone using hand gestures in real time. The project explores computer vision and deep learning techniques for safe and intuitive human–drone interaction.

---

## Overview

The system processes a live camera feed to detect hand gestures and map them to drone actions. MediaPipe is used for real-time hand detection and landmark extraction, while a custom CNN is trained on a self-collected dataset to classify directional gestures.

Drone commands are currently implemented as dummy functions for testing and demonstration.

---

## Tech Stack

- Python  
- OpenCV  
- MediaPipe (Tasks API)  
- TensorFlow / Keras  

---

## Gesture Recognition Pipeline

1. Webcam input  
2. MediaPipe hand detection and cropping  
3. CNN-based gesture classification  
4. State-based control logic  
5. Drone command execution (dummy)  

---

## Supported Gestures

### CNN-based gestures
- Stop  
- Up  
- Down  
- Left  
- Right  

### System-level gestures
- Photo capture with countdown  
- Follow-me mode using face detection  
- Safe return-to-user and shutdown sequence  

---

## Dataset

- Dataset is self-collected using a webcam  
- Hand images are cropped using MediaPipe during capture  
- Dataset split: 80% training, 20% validation  

---


---

## Notes

- The project focuses on perception and control logic rather than hardware integration.  
- All drone-related commands are placeholders.  

---

## Future Improvements

- Add more gesture classes  
- Improve robustness using data augmentation  
- Temporal gesture recognition  
- Edge deployment  
- Integration with real drone hardware  

---

## Author

Arnav Bisht  
AIMS DTU
