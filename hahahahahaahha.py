import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load processed data
with open("asl_A_data.pickle", "rb") as f:
    dataset = pickle.load(f)

data = np.array(dataset["data"])
labels = np.array(dataset["labels"])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)

# Train Random Forest classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained! Accuracy: {accuracy * 100:.2f}%")

# Save trained model
with open("asl_A_model.pickle", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as 'asl_A_model.pickle'")
