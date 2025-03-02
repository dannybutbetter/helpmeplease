from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import pickle
import base64

# Initialize Flask app
app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Load the trained model
with open("asl_model.pickle", "rb") as f:
    model = pickle.load(f)

# Create an app factory function
def create_app():
    return app

# Define ASL letters (Skipping J & Z)
asl_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

# ✅ Debugging: Print all requests
@app.before_request
def log_request():
    print(f"📥 Received request: {request.method} {request.path}")

# ✅ Home Route: Test if Flask is running
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "ASL Flask API is running!"})

# ✅ Prediction Route: Classifies ASL letter
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the image from the request
        image_data = request.json.get('image')
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image with MediaPipe
        results = hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return jsonify({'error': 'No hand detected in the image'})
        
        # Extract landmarks
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks.append(landmark.x)
                landmarks.append(landmark.y)
        
        # Make prediction
        prediction = model.predict([landmarks])
        
        # Convert prediction to letter (assuming same order as in create_dataset.py)
        predicted_letter = asl_letters[prediction[0]]
        
        return jsonify({
            'prediction': predicted_letter,
            'confidence': float(max(model.predict_proba([landmarks])[0]))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

# ✅ Run Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
