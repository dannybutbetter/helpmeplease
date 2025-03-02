from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
try:
    with open("asl_AZ_model.pickle", "rb") as f:
        model = pickle.load(f)
    print("✅ Model Loaded Successfully!")
except FileNotFoundError:
    print("❌ ERROR: Model file 'asl_AZ_model.pickle' not found!")
    model = None

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
def predict_asl():
    try:
        # Debug: Print request data
        print(f"📥 Raw Data: {request.data}")
        print(f"📥 JSON Data: {request.get_json()}")

        # Parse JSON input
        data = request.get_json()
        landmarks = data.get("landmarks", [])

        # Validate input size
        if len(landmarks) != 42:
            return jsonify({"error": "Invalid input size. Expected 42 values."}), 400

        # Convert to NumPy array
        landmarks = np.array(landmarks).reshape(1, -1)

        # Predict ASL letter
        prediction = model.predict(landmarks)[0]
        predicted_letter = asl_letters[prediction]

        return jsonify({"letter": predicted_letter})

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

# ✅ Run Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
