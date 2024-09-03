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

    if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return img



while True:
    _, img = cam.read()
    img = cv.flip(img, 1)

    img = findHands(img)

    cv.imshow("Camera", img)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break