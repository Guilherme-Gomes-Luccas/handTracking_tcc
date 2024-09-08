import cv2 as cv
import mediapipe as mp
import numpy as np
import math

# Inicializando as listas com dois espaços para as mãos
flexion = [None, None]  
proximity = [None, None]
contact = [None, None]
thumbDirection = [None, None]
palm = [None, None]
handDirection = [None, None]

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


def fingerFlexion(landmarks, handSide):
    fingerFlexion = []

    # Dedo polegar
    if handSide == "Right":
        if landmarks.landmark[4].x > landmarks.landmark[3].x:
            fingerFlexion.append(-1)  # Flexionado
        elif landmarks.landmark[4].x < landmarks.landmark[3].x:
            fingerFlexion.append(1)  # Não flexionado
        else:
            fingerFlexion.append(0)  # Neutro
    else:
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

def fingerProximity(landmarks, thresholdTogether=0.05, thresholdSeparated = 0.07):
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

def thumbContact(landmarks, thresholdTogether=0.06, thresholdSeparated = 0.10):
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

def palmDirection(landmarks):
    # One-hot encoding for palm orientation: Left, Right, Down, Up, Inward, Outward
    palmDirections = [0, 0, 0, 0, 0, 0] 
    
    # Coordenadas principais para definir os vetores
    palm = np.array([landmarks.landmark[0].x, landmarks.landmark[0].y, landmarks.landmark[0].z])  # Centro da palma
    base_middle_finger = np.array([landmarks.landmark[9].x, landmarks.landmark[9].y, landmarks.landmark[9].z])  # Base do dedo médio
    base_pinky_finger = np.array([landmarks.landmark[17].x, landmarks.landmark[17].y, landmarks.landmark[17].z])  # Base do dedo mínimo

    # Vetores que definem a orientação da mão
    palm_vector = base_middle_finger - palm
    base_vector = base_pinky_finger - palm

    # Vetor normal à palma da mão (produto vetorial dos dois vetores da palma)
    normal_vector = np.cross(palm_vector, base_vector)

    # Direção Z -> Determina se a palma está voltada para cima (Up) ou para baixo (Down)
    if normal_vector[2] > 0:
        palmDirections[3] = 1  # Up
    else:
        palmDirections[2] = 1  # Down

    # Direção X -> Determina se a palma está orientada para esquerda (Left) ou direita (Right)
    if normal_vector[0] > 0:
        palmDirections[1] = 1  # Right
    else:
        palmDirections[0] = 1  # Left

    # Direção Y -> Determina se a palma está voltada para dentro (Inward) ou para fora (Outward)
    if normal_vector[1] > 0:
        palmDirections[5] = 1  # Outward
    else:
        palmDirections[4] = 1  # Inward

    return palmDirections

def handLandmarks(landmarks):
    return [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]

def process_hand_data(hand_data, index):
    flexion[index] = {
        'info': hand_data['info'].classification[0].label,
        'value': fingerFlexion(hand_data['landmarks'], hand_data['info'].classification[0].label)
    }

    proximity[index] = {
        'info': hand_data['info'].classification[0].label,
        'value': fingerProximity(hand_data['landmarks'])
    }

    contact[index] = {
        'info': hand_data['info'].classification[0].label,
        'value': thumbContact(hand_data['landmarks'])
    }

    thumbDirection[index] = {
        'info': hand_data['info'].classification[0].label,
        'value': pointingThumbDirection(hand_data['landmarks'])
    }

    palm[index] = {
        'info': hand_data['info'].classification[0].label,
        'value': palmDirection(hand_data['landmarks'])
    }

    handDirection[index] = {
        'info': hand_data['info'].classification[0].label,
        'values': handLandmarks(hand_data['landmarks'])
    }

while True:
    ret, img = cam.read()
    if not ret:
        break

    img = cv.flip(img, 1)
    img, hands_data = findHands(img)

    for i, hand_data in enumerate(hands_data[:2]):  # Limita ao máximo de 2 mãos
        process_hand_data(hand_data, i)

    print(handDirection)

    # Mostra a imagem da câmera
    cv.imshow("Camera", img)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# Libera a câmera e fecha as janelas
cam.release()
cv.destroyAllWindows()