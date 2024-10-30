import cv2 as cv
import mediapipe as mp
import numpy as np
import math

# Inicializando as listas com dois espaços para as mãos
flexion = [None, None]  
proximity = [None, None]
fingerThumbContact = [None, None]
thumbDirection = [None, None]
palm = [None, None]
handOrientation = [None, None]
handDirection = [None, None]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True,  # Modo de imagem estática
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# Carregue a imagem fornecida
image_path = "D:\\TCC\\okSign\\okSign1.jpeg"
img = cv.imread(image_path)

# Função para processar e identificar as mãos na imagem
def findHands(img):
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    hands_data = []

    if result.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hands_data.append({'landmarks': hand_landmarks, 'info': hand_info})

    return img, hands_data

# Função para calcular a distância euclidiana entre dois pontos
def euclidean_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + 
                     (point1.y - point2.y) ** 2 + 
                     (point1.z - point2.z) ** 2)

def calculate_angle(a, b, c):
    ab = np.array([b.x - a.x, b.y - a.y, b.z - a.z])
    bc = np.array([b.x - c.x, b.y - c.y, b.z - c.z])

    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0) 
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# Função para checar se os dedos estão flexionados usando os ângulos
def fingerFlexion(landmarks, handSide):
    fingerFlexion = []

    # Polegar
    thumb_angle = calculate_angle(landmarks.landmark[2], landmarks.landmark[3], landmarks.landmark[4])
    if thumb_angle < 70:
        fingerFlexion.append(-1)  # Dobrado
    elif thumb_angle > 150:
        fingerFlexion.append(1)  # Reto
    else:
        fingerFlexion.append(0)  # Incerto

    # Outros dedos
    for finger_points in [[5, 6, 8], [9, 10, 12], [13, 14, 16], [17, 18, 20]]:
        angle = calculate_angle(landmarks.landmark[finger_points[0]],
                                landmarks.landmark[finger_points[1]],
                                landmarks.landmark[finger_points[2]])
        if angle < 100:
            fingerFlexion.append(-1)  # Dobrado
        elif angle > 160:
            fingerFlexion.append(1)  # Reto
        else:
            fingerFlexion.append(0)  # Incerto

    return fingerFlexion

# Função que checa a proximidade dos dedos
def fingerProximity(landmarks, thresholdTogether=0.05, thresholdSeparated=0.07):
    fingerProximity = []

    for tip, dip, pip in zip([8, 12, 16], [7, 11, 15], [6, 10, 14]):
        distance = []
        distance.append(euclidean_distance(landmarks.landmark[tip], landmarks.landmark[tip+4]))
        distance.append(euclidean_distance(landmarks.landmark[dip], landmarks.landmark[dip+4]))
        distance.append(euclidean_distance(landmarks.landmark[pip], landmarks.landmark[pip+4]))

        avgDistance = np.mean(distance)

        if avgDistance < thresholdTogether:
            fingerProximity.append(1)
        elif avgDistance > thresholdSeparated:
            fingerProximity.append(-1)
        else:
            fingerProximity.append(0)

    return fingerProximity

# Função para detectar o contato dos dedos com o polegar
def thumbContact(landmarks, thresholdTogether=0.075, thresholdSeparated=0.10):
    thumbContact = []

    for tip in [8, 12, 16, 20]:
        distance = euclidean_distance(landmarks.landmark[4], landmarks.landmark[tip])

        print(f"{distance} < {thresholdTogether} = {distance <= thresholdTogether}")

        if distance <= thresholdTogether:
            thumbContact.append(1)
        elif distance >= thresholdSeparated:
            thumbContact.append(-1)
        else:
            thumbContact.append(0)

    return thumbContact

def verifyThumbContact(value):
    if value == 1:
        return True
    return False

# Função para detectar a direção do polegar
def pointingThumbDirection(landmarks, handSide):
    if handSide == fingerThumbContact[0]['info']:
        values = fingerThumbContact[0]['value']
    else:
        values = fingerThumbContact[1]['value']
    
    for i in values:
        print(i)
        if verifyThumbContact(i):
            return 0

    tip = landmarks.landmark[4]   # Ponta do polegar
    base = landmarks.landmark[3]  # Base do polegar
    wrist = landmarks.landmark[0] # Pulso

    if tip.y < base.y and tip.y < wrist.y:
        return 1  # Polegar para cima
    
    elif tip.y > base.y and tip.y > wrist.y:
        return -1  # Polegar para baixo

    delta_x = tip.x - base.x
    delta_y = tip.y - base.y

    angle = math.atan2(delta_y, delta_x)  # Ângulo em radianos

    if abs(angle) < 0.2 or abs(abs(angle) - math.pi) < 0.2:
        return 0  # Polegar de lado

# Função para detectar a direção da mão
def palmDirection(landmarks):
    palmDirections = [0, 0, 0, 0, 0, 0] 
    
    palm = np.array([landmarks.landmark[0].x, landmarks.landmark[0].y, landmarks.landmark[0].z])  
    base_middle_finger = np.array([landmarks.landmark[9].x, landmarks.landmark[9].y, landmarks.landmark[9].z])  
    base_pinky_finger = np.array([landmarks.landmark[17].x, landmarks.landmark[17].y, landmarks.landmark[17].z])  

    palm_vector = base_middle_finger - palm
    base_vector = base_pinky_finger - palm

    normal_vector = np.cross(palm_vector, base_vector) # cálculo do vetor normal

    if normal_vector[2] > 0:
        palmDirections[3] = 1  # Up
    else:
        palmDirections[2] = 1  # Down

    if normal_vector[0] > 0:
        palmDirections[1] = 1  # Right
    else:
        palmDirections[0] = 1  # Left

    if normal_vector[1] > 0:
        palmDirections[5] = 1  # Outward
    else:
        palmDirections[4] = 1  # Inward

    return palmDirections

def handOrientationPosition(landmarks, handSide):
    # Ponto 0 (base da palma) e ponto 9 (meio do dedo médio)
    basePalm = landmarks.landmark[0]
    baseMiddleFinger = landmarks.landmark[9]

    deltaX = baseMiddleFinger.x - basePalm.x
    deltaY = baseMiddleFinger.y - basePalm.y

    if deltaX == 0:
        return 0  # Inclinação infinita

    inclination = deltaY / deltaX

    if handSide == "Left":
        inclination = -inclination

    if abs(inclination) > 1:
        return 1  # Mão vertical
    else:
        return -1  # Mão horizontal

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

    fingerThumbContact[index] = {
        'info': hand_data['info'].classification[0].label,
        'value': thumbContact(hand_data['landmarks'])
    }

    thumbDirection[index] = {
        'info': hand_data['info'].classification[0].label,
        'value': pointingThumbDirection(hand_data['landmarks'], hand_data['info'].classification[0].label)
    }

    palm[index] = {
        'info': hand_data['info'].classification[0].label,
        'value': palmDirection(hand_data['landmarks'])
    }

    handOrientation[index] = {
        'info': hand_data['info'].classification[0].label,
        'value': handOrientationPosition(hand_data['landmarks'], hand_data['info'].classification[0].label)
    }

    handDirection[index] = {
        'info': hand_data['info'].classification[0].label,
        'values': handLandmarks(hand_data['landmarks'])
    }

# Processa a imagem
img, hands_data = findHands(img)

for i, hand_data in enumerate(hands_data[:2]):
    process_hand_data(hand_data, i)

print(f"Finger Flexion: {flexion}")
print(f"Finger Proximity: {proximity}")
print(f"Finger in contact with Thumb: {fingerThumbContact}")
print(f"Thumb Direction: {thumbDirection}")
print(f"Palm orientation: {palm}")
print(f"Hand Orientation: {handOrientation}")
print(f"Hand Coordinates: {handDirection}")

# Mostra a imagem processada com as marcações
cv.imshow("Image", img)
cv.waitKey(0)
cv.destroyAllWindows()