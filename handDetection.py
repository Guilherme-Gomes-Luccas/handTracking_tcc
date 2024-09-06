import cv2 as cv
import mediapipe as mp
import numpy as np
import math

# Inicializando as listas com dois espaços para as mãos
flexion = [None, None]  
proximity = [None, None]
contact = [None, None]
thumbDirection = [None, None]

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


def fingerFlexion(landmarks):
    fingerFlexion = []

    # Dedo polegar
    if landmarks.landmark[4].x > landmarks.landmark[3].x:
        fingerFlexion.append(1)  # Flexionado
    elif landmarks.landmark[4].x < landmarks.landmark[3].x:
        fingerFlexion.append(-1)  # Não flexionado
    else:
        fingerFlexion.append(0)  # Neutro

    # Outros dedos
    for tip in [8, 12, 16, 20]:
        if landmarks.landmark[tip].y < landmarks.landmark[tip - 2].y:
            fingerFlexion.append(1)  # Flexionado
        elif landmarks.landmark[tip].y > landmarks.landmark[tip - 2].y:
            fingerFlexion.append(-1)  # Não flexionado
        else:
            fingerFlexion.append(0)  # Neutro

    return fingerFlexion

# Função para calcular a distância euclidiana entre dois pontos
def euclidean_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + 
                     (point1.y - point2.y) ** 2 + 
                     (point1.z - point2.z) ** 2)

def fingerProximity(landmarks, thresholdTogether=0.05, thresholdSeparated = 0.10):
    fingerProximity = []

    for tip, nextPoint in zip([8, 12, 15], [11, 16, 20]):
        distance = euclidean_distance(landmarks.landmark[tip], landmarks.landmark[nextPoint])

        if distance < thresholdTogether:
            fingerProximity.append(1)
        elif distance > thresholdSeparated:
            fingerProximity.append(-1)
        else:
            fingerProximity.append(0)

    return fingerProximity

def thumbContact(landmarks, thresholdTogether=0.08, thresholdSeparated = 0.11):
    thumbContact = []

    for tip in [8, 12, 16, 20]:
        distance = euclidean_distance(landmarks.landmark[4], landmarks.landmark[tip])

        if distance < thresholdTogether:
            thumbContact.append(1)
        elif distance > thresholdSeparated:
            thumbContact.append(-1)
        else:
            thumbContact.append(0)

    return thumbContact

import math

def pointingThumbDirection(landmarks, sideThreshold=10):
    tip = landmarks.landmark[4]   # Ponta do polegar
    base = landmarks.landmark[3]  # Base do polegar
    wrist = landmarks.landmark[0] # Pulso

    # Verifica se o polegar está para cima (tip acima da base e do pulso)
    if tip.y < base.y and tip.y < wrist.y:
        return 1  # Polegar para cima
    
    # Verifica se o polegar está para baixo (tip abaixo da base e do pulso)
    elif tip.y > base.y and tip.y > wrist.y:
        return -1  # Polegar para baixo

    # Calcular o vetor entre a base e a ponta do polegar
    delta_x = tip.x - base.x
    delta_y = tip.y - base.y

    # Calcular a inclinação (ângulo) do vetor
    angle = math.atan2(delta_y, delta_x)  # Ângulo em radianos

    # Verifica se o polegar está de lado (ângulo próximo de 0 ou π)
    if abs(angle) < sideThreshold or abs(abs(angle) - math.pi) < sideThreshold:
        return 0  # Polegar de lado

    # Caso não se encaixe nas condições acima, podemos retornar um valor padrão
    return None

def euclidean_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + 
                     (point1.y - point2.y) ** 2 + 
                     (point1.z - point2.z) ** 2)



while True:
    ret, img = cam.read()
    if not ret:
        break

    img = cv.flip(img, 1)
    img, hands_data = findHands(img)

    if len(hands_data) > 0:
        flexion[0] = {
            'info': hands_data[0]['info'].classification[0].label,
            'value': fingerFlexion(hands_data[0]['landmarks'])
        }

        proximity[0] = {
            'info': hands_data[0]['info'].classification[0].label,
            'value': fingerProximity(hands_data[0]['landmarks'])
        }

        contact[0] = {
            'info': hands_data[0]['info'].classification[0].label,
            'value': thumbContact(hands_data[0]['landmarks'])
        }

        thumbDirection[0] = {
            'info': hands_data[0]['info'].classification[0].label,
            'value': pointingThumbDirection(hands_data[0]['landmarks'])
        }

        print(thumbDirection[0])

        if len(hands_data) > 1:
            flexion[1] = {
                'info': hands_data[1]['info'].classification[0].label,
                'value': fingerFlexion(hands_data[1]['landmarks'])
            }

            proximity[1] = {
                'info': hands_data[1]['info'].classification[0].label,
                'value': fingerProximity(hands_data[1]['landmarks'], 0.06, 0.13)
            }

            contact[1] = {
                'info': hands_data[1]['info'].classification[0].label,
                'value': thumbContact(hands_data[1]['landmarks'])
            }


    #Exibe a flexão detectada
    #if flexion[0] is not None:
        #print(f'Mão 1: {flexion[0]}')
    #if flexion[1] is not None:
        #print(f'Mão 2: {flexion[1]}')

    # Mostra a imagem da câmera
    cv.imshow("Camera", img)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# Libera a câmera e fecha as janelas
cam.release()
cv.destroyAllWindows()