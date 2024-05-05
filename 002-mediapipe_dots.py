import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1)

cap = cv2.VideoCapture(0)

tip_ids = [4, 8, 12, 16, 20]
base_ids = [0, 5, 9, 13, 17]
extension_threshold = 0.17

def get_vector(p1, p2):
    return np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])

def is_finger_extended(base, tip, is_thumb=False):
    base_to_tip = get_vector(base, tip)
    base_to_tip_norm = np.linalg.norm(base_to_tip)
    return base_to_tip_norm > extension_threshold
    
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,
                                      hand_landmarks,
                                      mp_hands.HAND_CONNECTIONS)
        if hand_landmarks:
            landmarks = hand_landmarks.landmark
            for id, landmark in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.putText(frame,
                            str(id),
                            (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            1)
                
            for finger_index, tip_id in enumerate(tip_ids):
                base_id = base_ids[finger_index]  
                if is_finger_extended(landmarks[base_id], landmarks[tip_id]):
                    cx, cy = int(landmarks[tip_id].x * frame.shape[1]), int(landmarks[tip_id].y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)


    cv2.imshow('Fingers', frame)

    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()
    
