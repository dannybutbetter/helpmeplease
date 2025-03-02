import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load processed data
with open("asl_AZ_data.pickle", "rb") as f:
    dataset = pickle.load(f)

data = np.array(dataset["data"])
labels = np.array(dataset["labels"])

# Normalize data (scale all landmark values between 0 and 1)
data = (data - np.min(data)) / (np.max(data) - np.min(data))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)

# Train MLP Neural Network
model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),  # 3-layer neural network
    activation='relu',
    solver='adam',
    batch_size=32,
    max_iter=500  # Train for more epochs
)
model.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Neural Network Model Accuracy: {accuracy * 100:.2f}%")

# Save trained model
with open("asl_AZ_model.pickle", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as 'asl_AZ_model.pickle'")

# ----------------- REAL-TIME DETECTION -----------------
# Load trained model
with open("asl_AZ_model.pickle", "rb") as f:
    model = pickle.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5)

# Define letters used in training (A-Y, skipping J and Z)
asl_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

# Create dictionary mapping predictions to letters
labels_dict = {i: asl_letters[i] for i in range(len(asl_letters))}

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append(landmark.x)
                landmarks.append(landmark.y)

            # Predict using model
            prediction = model.predict([landmarks])[0]
            label_text = labels_dict.get(prediction, "Unknown")

            # Display the predicted letter
            cv2.putText(frame, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Detection (A-Y, No J or Z)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
