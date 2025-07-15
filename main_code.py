import cv2
from keras.models import model_from_json
import numpy as np

# ------------------- Load Emotion Detection Model -------------------
with open("facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()
emotion_model = model_from_json(model_json)
emotion_model.load_weights("facialemotionmodel.h5")

# ------------------- Load Gender Detection Model -------------------
gender_model = cv2.dnn.readNetFromCaffe(
    "models/deploy_gender.prototxt",
    "models/gender_net.caffemodel"
)

# ------------------- Constants -------------------
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
gender_labels = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# ------------------- Face Detector -------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ------------------- Preprocessing Function -------------------
def extract_emotion_features(image):
    feature = np.array(image).reshape(1, 48, 48, 1)
    return feature / 255.0

# ------------------- Start Webcam -------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract face regions
        face_color = frame[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]

        # ------------ Emotion Detection ------------
        try:
            face_emotion = cv2.resize(face_gray, (48, 48))
            emotion_input = extract_emotion_features(face_emotion)
            emotion_pred = emotion_model.predict(emotion_input)
            emotion = emotion_labels[np.argmax(emotion_pred)]
        except:
            emotion = "N/A"

        # ------------ Gender Detection ------------
        try:
            face_resized = cv2.resize(face_color, (227, 227))
            blob = cv2.dnn.blobFromImage(face_resized, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_model.setInput(blob)
            gender_pred = gender_model.forward()
            gender = gender_labels[gender_pred[0].argmax()]
        except:
            gender = "N/A"

        # ------------ Display Results ------------
        label = f"{gender}, {emotion}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Emotion and Gender Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
        break

cap.release()
cv2.destroyAllWindows()
