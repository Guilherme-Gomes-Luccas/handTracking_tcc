import cv2 as cv
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)



cam = cv.VideoCapture(0)

cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

def findHands(img):
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    result = hands.process(img_rgb)

    hands_data = []

    if result.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hands_data.append({'landmarks': hand_landmarks, 'info': hand_info})

    return img, hands_data


def fingerFlection(landmarks):
     flection = []

     #print(landmarks.landmark[8].y)

     for tip in [8, 12, 16, 20]:
            if landmarks.landmark[tip].y < landmarks.landmark[tip-2].y:
                 print("ponta {tip} levantada")
                 

     return True


while True:

    _, img = cam.read()
    img = cv.flip(img, 1)

    img, hands_data = findHands(img)

    if len(hands_data) > 0:
        fingerFlection(hands_data[0]['landmarks'])
    
    #    print(hands_data[0]['info'])

    cv.imshow("Camera", img)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break