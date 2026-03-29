import cv2
import numpy as np
import torch
from torchvision import transforms
from model import EmotionResNet
from utils import load_model
from collections import deque

MODEL_path = "models/emotion_resnet.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

model = EmotionResNet(num_classes=7)
model = load_model(model, MODEL_path, device)

AGE_MODEL = "models/age_net.caffemodel"
AGE_PROTO = "models/age_deploy.prototxt"
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)

age_list = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(21-24)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

GENDER_MODEL = "models/gender_net.caffemodel"
GENDER_PROTO = "models/gender_deploy.prototxt"
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

gender_list = ["Male", "Female"]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,), (0.229,))
    ])

emotion_buffer = deque(maxlen=10)
conf_buffer = deque(maxlen=10)

color_map = {
    "happy": (0, 255, 0),
    "sad": (255, 0, 0),
    "angry": (0, 0, 255),
    "surprise": (0, 255, 255),
    "neutral": (200, 200, 200),
    "fear": (255, 0, 255),
    "disgust": (0, 128, 128)
}

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        try:
            img = transform(face).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)

            label = classes[pred.item()]
            confidence = conf.item()
            conf_buffer.append(confidence)
            
            emotion_buffer.append(label)
            conf_buffer.append(confidence)

            final_label = max(set(emotion_buffer), key=emotion_buffer.count)

            smooth_conf = sum(conf_buffer) / len(conf_buffer)
            confidence = smooth_conf * 100

            if confidence < 40:
                continue
            
            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]

            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            
            color = color_map.get(final_label, (0, 255, 0))
            text = f"{final_label} ({confidence:.1f}%)|{gender}|{age}"
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(
                frame,
                text,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        except:
            continue

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
