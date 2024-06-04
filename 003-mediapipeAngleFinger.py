import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk

# Получение размеров экрана
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

cap = cv2.VideoCapture(0)

# Установка разрешения видео в Full HD
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Устанавливаем размер окна на 80% от размера экрана
window_width = int(screen_width * 0.9)
window_height = int(screen_height * 0.9)

tip_ids = [4, 8, 12, 16, 20]
base_ids = [0, 5, 9, 13, 17]
joint_ids = [3, 6, 10, 14, 18]

# Пороговые значения углов для пальцев
thumb_bend_threshold = 40
finger_bend_threshold = 50

def get_angle(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cosine_angle = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def is_finger_bent(base, joint, tip, is_thumb=False):
    v1 = [joint.x - base.x, joint.y - base.y, joint.z - base.z]
    v2 = [tip.x - joint.x, tip.y - joint.y, tip.z - joint.z]
    angle = get_angle(v1, v2)
    if is_thumb:
        return angle < thumb_bend_threshold
    else:
        return angle < finger_bend_threshold

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            
            for id, landmark in enumerate(landmarks):
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            for finger_index, tip_id in enumerate(tip_ids):
                base_id = base_ids[finger_index]
                joint_id = joint_ids[finger_index]
                is_thumb = (finger_index == 0)
                if is_finger_bent(landmarks[base_id], landmarks[joint_id], landmarks[tip_id], is_thumb):
                    cx, cy = int(landmarks[tip_id].x * frame.shape[1]), int(landmarks[tip_id].y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
                else:
                    cx, cy = int(landmarks[tip_id].x * frame.shape[1]), int(landmarks[tip_id].y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    # Масштабируем изображение до размера окна
    frame_resized = cv2.resize(frame, (window_width, window_height))
    
    cv2.imshow('Fingers', frame_resized)
    cv2.resizeWindow('Fingers', window_width, window_height)

    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()
