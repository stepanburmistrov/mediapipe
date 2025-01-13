import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

insert_image = cv2.imread("image.jpg")

h, w = insert_image.shape[:2]
src_points = np.array([
    [0, 0],
    [w - 1, 0],
    [w - 1, h - 1],
    [0, h - 1]
], dtype="float32")

cap = cv2.VideoCapture(0)

while True:
    read_ok, frame = cap.read()
    if not read_ok:
        break
    frame = cv2.flip(frame,1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    x4L,y4L,x8L,y8L = None, None, None, None
    x4R,y4R,x8R,y8R = None, None, None, None
    result = None
    
    if results.multi_hand_landmarks is not None:
        for hand_landmark, hand_nandedness in zip(results.multi_hand_landmarks,results.multi_handedness):
            for point_id, landmark in enumerate(hand_landmark.landmark):
                h, w, c = frame.shape
                point_x, point_y = int(landmark.x * w), int(landmark.y * h)
                label = hand_nandedness.classification[0].label
                if point_id == 4:
                    if label == "Right":
                        x4R,y4R = point_x, point_y
                    if label == "Left":
                        x4L,y4L = point_x, point_y
                    
                if point_id == 8:
                    if label == "Right":
                        x8R,y8R = point_x, point_y
                    if label == "Left":
                        x8L,y8L = point_x, point_y
                                                
                cv2.circle(frame, (point_x,point_y), 2, (0,255,0), -1)
                cv2.putText(frame, str(point_id), (point_x,point_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255),1)
                
            if x4L is not None and x4R is not None:
                cv2.line(frame, (x8L,y8L), (x8R,y8R), (0,255,0), 5)
                cv2.line(frame, (x4L,y4L), (x4R,y4R), (0,255,0), 5)
                cv2.line(frame, (x4L,y4L), (x8L,y8L), (0,255,0), 5)
                cv2.line(frame, (x4R,y4R), (x8R,y8R), (0,255,0), 5)
                dst_points = np.array([
                                        (x8L,y8L),
                                        (x8R,y8R),
                                        (x4R,y4R),
                                        (x4L,y4L) 
                                    ], dtype="float32")

                matrix = cv2.getPerspectiveTransform(src_points, dst_points)


                warped_insert = cv2.warpPerspective(insert_image,
                                                    matrix,
                                                    (frame.shape[1],
                                                     frame.shape[0]))
                
                cv2.imwrite("warped.jpg", warped_insert)
                mask = np.zeros_like(frame)
                cv2.fillPoly(mask, [np.int32(dst_points)], (255, 255, 255))
                cv2.imwrite("mask.jpg", mask)
                mask_inv = cv2.bitwise_not(mask)
                main_image_bg = cv2.bitwise_and(frame, mask_inv)
                cv2.imwrite("main_image_bg.jpg", main_image_bg)
                result = cv2.add(main_image_bg, warped_insert)
                cv2.imwrite("result.jpg", result)

            
    cv2.imshow("camera", result if result is not None else frame)
    k = cv2.waitKey(30)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
