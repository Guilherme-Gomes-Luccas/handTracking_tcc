import cv2 as cv
import mediapipe as mp
import numpy as np

mpHands = mp.solutions.hands
hands = mpHands.Hands()

cam = cv.VideoCapture(0)

cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    _, img = cam.read()

    cv.imshow("Camera", img)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break