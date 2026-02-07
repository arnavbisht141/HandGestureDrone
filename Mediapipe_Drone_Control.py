import cv2
import time
import numpy as np
from enum import Enum
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ===================== FACE DETECTOR =====================
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError(
        "Failed to load haarcascade_frontalface_default.xml"
    )

# ===================== SYSTEM STATE ====================
class SystemState(Enum):
    NORMAL = 0
    PHOTO_COUNTDOWN = 1
    SHUTDOWN_CONFIRM = 2
    RETURN_TO_USER = 3
    FOLLOW_ME = 4  

state = SystemState.NORMAL
state_start = None

follow_hold_start = None
FOLLOW_HOLD = 2

# ===================== CONSTANTS =====================
PHOTO_DELAY = 3
SHUTDOWN_HOLD = 3
RETURN_TIME = 10

MIN_SPEED = 0.5
MAX_SPEED = 3.0
DEFAULT_SPEED = 0.0
current_speed = DEFAULT_SPEED

# ===================== DRONE COMMANDS (DUMMY) =====================
def stop(): print("🛑 STOP")
def go_up(v): print(f"⬆️ UP @ {v:.2f} m/s")
def go_down(v): print(f"⬇️ DOWN @ {v:.2f} m/s")
def go_left(v): print(f"⬅️ LEFT @ {v:.2f} m/s")
def go_right(v): print(f"➡️ RIGHT @ {v:.2f} m/s")
def go_away(v): print(f"↗️ AWAY @ {v:.2f} m/s")
def come_near(v): print(f"↘️ NEAR @ {v:.2f} m/s")
def barrel_roll(): print("🔄 BARREL ROLL")
def shutdown(): print("🛑 SYSTEM SHUTDOWN")

# ===================== GESTURE HELPERS =====================
def finger_up(lm, tip, pip):
    return lm[tip][1] < lm[pip][1]

def thumb_up(lm, handed):
    return lm[4][0] > lm[3][0] if handed == "Right" else lm[4][0] < lm[3][0]

def finger_states(lm, handed):
    return (
        thumb_up(lm, handed),
        finger_up(lm, 8, 6),
        finger_up(lm, 12, 10),
        finger_up(lm, 16, 14),
        finger_up(lm, 20, 18)
    )

def detect_right_hand_gesture(lm):
    t,i,m,r,p = finger_states(lm, "Right")
    if all([t,i,m,r,p]): return "STOP"
    if i and not any([t,m,r,p]): return "UP"
    if i and m and not any([t,r,p]): return "DOWN"
    if p and not any([t,i,m,r]): return "LEFT"
    if t and not any([i,m,r,p]): return "RIGHT"
    if i and m and r and not any([t,p]): return "AWAY"
    if i and m and r and p and not t: return "NEAR"
    if t and i and p and not m and not r: return "ROLL"
    return None

def peace_sign(states):
    t,i,m,r,p = states
    return i and m and not any([t,r,p])

def open_palm(states):
    return all(states)

def follow_gesture(states):
    t,i,m,r,p = states
    return t and i and p and not m and not r

# ===================== SPEED CONTROL =====================
def compute_speed(left_lm):
    d = np.linalg.norm(left_lm[4][:2] - left_lm[8][:2])
    d = np.clip(d, 0.02, 0.25)
    return np.interp(d, [0.02, 0.25], [MIN_SPEED, MAX_SPEED])

# ===================== MEDIAPIPE TASKS =====================
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

print("[INFO] ESC to quit")

# ===================== MAIN LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    clean_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    right_hand, left_hand = None, None

    if result.hand_landmarks:
        for i, lm in enumerate(result.hand_landmarks):
            handed = result.handedness[i][0].category_name
            pts = np.array([[p.x, p.y, p.z] for p in lm])

            for p in pts:
                cv2.circle(
                    frame,
                    (int(p[0]*frame.shape[1]), int(p[1]*frame.shape[0])),
                    4, (0,255,0), -1
                )

            if handed == "Right":
                right_hand = pts
            else:
                left_hand = pts

    now = time.time()
    both_hands = right_hand is not None and left_hand is not None

    # ===================== FOLLOW-ME TOGGLE =====================
    if state == SystemState.NORMAL and both_hands:
        rs = finger_states(right_hand, "Right")
        ls = finger_states(left_hand, "Left")

        if follow_gesture(rs) and follow_gesture(ls):
            if follow_hold_start is None:
                follow_hold_start = now
            elif now - follow_hold_start >= FOLLOW_HOLD:
                state = SystemState.FOLLOW_ME
                follow_hold_start = None
        else:
            follow_hold_start = None

    elif state == SystemState.FOLLOW_ME and both_hands:
        rs = finger_states(right_hand, "Right")
        ls = finger_states(left_hand, "Left")

        if follow_gesture(rs) and follow_gesture(ls):
            if follow_hold_start is None:
                follow_hold_start = now
            elif now - follow_hold_start >= FOLLOW_HOLD:
                state = SystemState.NORMAL
                follow_hold_start = None
        else:
            follow_hold_start = None

    # ===================== NORMAL =====================
    if state == SystemState.NORMAL:

        current_speed = (
            compute_speed(left_hand)
            if left_hand is not None else DEFAULT_SPEED
        )

        if both_hands:
            rs = finger_states(right_hand, "Right")
            ls = finger_states(left_hand, "Left")

            if peace_sign(rs) and peace_sign(ls):
                state = SystemState.PHOTO_COUNTDOWN
                state_start = now

            elif open_palm(rs) and open_palm(ls):
                state = SystemState.SHUTDOWN_CONFIRM
                state_start = now

        if state == SystemState.NORMAL and right_hand is not None:
            gesture = detect_right_hand_gesture(right_hand)

            if gesture == "STOP": stop()
            elif gesture == "UP": go_up(current_speed)
            elif gesture == "DOWN": go_down(current_speed)
            elif gesture == "LEFT": go_left(current_speed)
            elif gesture == "RIGHT": go_right(current_speed)
            elif gesture == "AWAY": go_away(current_speed)
            elif gesture == "NEAR": come_near(current_speed)
            elif gesture == "ROLL": barrel_roll()

            cv2.putText(frame, f"CMD: {gesture}",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,255,0), 2)

    # ===================== FOLLOW-ME =====================
    elif state == SystemState.FOLLOW_ME:
        cv2.putText(frame, "FOLLOW MODE ACTIVE",
                    (30,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (255,255,0), 3)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
            cx = x + w//2
            if cx < frame.shape[1]*0.4:
                print("↺ YAW LEFT")
            elif cx > frame.shape[1]*0.6:
                print("↻ YAW RIGHT")

    # ===================== PHOTO =====================
    elif state == SystemState.PHOTO_COUNTDOWN:
        rem = PHOTO_DELAY - int(now - state_start)
        cv2.putText(frame, f"PHOTO IN {rem}",
                    (30,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (0,255,255), 3)
        if rem <= 0:
            cv2.imwrite(f"photo_{int(now)}.jpg", clean_frame)
            state = SystemState.NORMAL

    # ===================== SHUTDOWN CONFIRM =====================
    elif state == SystemState.SHUTDOWN_CONFIRM:
        if not both_hands:
            state = SystemState.NORMAL
        else:
            rs = finger_states(right_hand, "Right")
            ls = finger_states(left_hand, "Left")
            if open_palm(rs) and open_palm(ls):
                rem = SHUTDOWN_HOLD - int(now - state_start)
                cv2.putText(frame, f"HOLD TO SHUTDOWN {rem}",
                            (30,80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                            (0,0,255), 3)
                if rem <= 0:
                    state = SystemState.RETURN_TO_USER
                    state_start = now
            else:
                state = SystemState.NORMAL

    # ===================== RETURN =====================
    elif state == SystemState.RETURN_TO_USER:
        cv2.putText(frame, "RETURNING TO USER",
                    (30,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (255,255,0), 3)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        if now - state_start >= RETURN_TIME:
            shutdown()
            break

    # ===================== UI =====================
    cv2.putText(frame, f"STATE: {state.name}",
                (20, frame.shape[0]-40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255,255,255), 2)

    cv2.putText(frame, f"SPEED: {current_speed:.2f} m/s",
                (20, frame.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0,255,255), 2)

    cv2.imshow("Drone Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

