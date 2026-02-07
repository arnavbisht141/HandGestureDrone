import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

# ===================== CONFIG =====================
MODEL_PATH = "gesture_cnn_baseline.h5"
IMG_SIZE = 128
CONF_THRESHOLD = 0.6   # ignore weak predictions

# MUST match training order
CLASS_NAMES = ['down', 'left', 'right', 'stop', 'up']

# ===================== LOAD CNN =====================
model = load_model(MODEL_PATH)
print("[INFO] CNN model loaded")

# ===================== MEDIAPIPE =====================
base = python.BaseOptions(model_asset_path="hand_landmarker.task")
opts = vision.HandLandmarkerOptions(
    base_options=base,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(opts)

# ===================== CAMERA =====================
cap = cv2.VideoCapture(0)
print("[INFO] Press ESC to quit")

# ===================== MAIN LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    display = frame.copy()

    # ---------- MediaPipe ----------
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    result = detector.detect(mp_image)
    crop = None

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in lm])

        x_min, y_min = np.min(pts, axis=0).astype(int)
        x_max, y_max = np.max(pts, axis=0).astype(int)

        pad = 20
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_max + pad)
        y_max = min(h, y_max + pad)

        if x_max > x_min and y_max > y_min:
            crop = frame[y_min:y_max, x_min:x_max]
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max),
                          (0,255,0), 2)

    # ---------- CNN INFERENCE ----------
    if crop is not None:
        img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img, verbose=0)[0]
        idx = np.argmax(preds)
        conf = preds[idx]

        if conf >= CONF_THRESHOLD:
            label = CLASS_NAMES[idx].upper()
            text = f"{label} ({conf:.2f})"
            color = (0,255,0)
        else:
            text = "UNSURE"
            color = (0,0,255)

        cv2.putText(display, text,
                    (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, color, 3)

        # optional preview
        cv2.imshow("Hand Crop", cv2.resize(crop, (IMG_SIZE, IMG_SIZE)))

    else:
        cv2.putText(display, "NO HAND",
                    (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0,0,255), 3)

    cv2.imshow("CNN Live Inference", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
