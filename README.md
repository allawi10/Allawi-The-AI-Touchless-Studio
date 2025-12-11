# AI Hand-Control System
An AI-powered system that allows users to control the computer using hand gestures through a webcam.  
This project uses MediaPipe Hands + OpenCV to enable touch-free control across multiple modes.

---

## 🚀 Features

### 1. AI Login System
- Recognizes hand-drawn numbers (0–9)
- Delete gesture
- Enter gesture
- Login works without a keyboard

### 2. Presentation Mode
- Next / previous slide using gestures  
- Smart pointer controlled by the index finger  
- Smooth real-time tracking

### 3. Image Processing Mode
- Rotate image  
- Change filters  
- Adjust brightness  
- Switch to mobile view  
- Navigate images with gestures

### 4. Whiteboard Mode
- Draw using index finger  
- Erase with all fingers  
- Real-time drawing

### 5. Geometry Mode
- Measure width/height between both hands  
- Real-time cm calculation  
- Uses thumb + index finger on both hands

### 6. Art Mode
- Interactive line effects  
- Lines move based on finger position  
- Creative motion visualization

---

## 🖐️ Gesture Controls

| Gesture | Action |
|--------|--------|
| Index finger | Move cursor / draw |
| All fingers | Erase |
| Thumb | Change filter |
| Little finger | Rotate image |
| Thumb + index (two hands) | Measure geometry |
| Finger count (0–10) | Login digits |

---

## 🧠 Technologies Used
- Python  
- OpenCV  
- MediaPipe  
- NumPy  
- PyAutoGUI  

---

## ▶️ How to Run

### 1. Install requirements
`bash
pip install opencv-python mediapipe numpy pyautogui
