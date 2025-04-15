# Load everything as before
import pickle
from datetime import datetime

with open("rf_parking_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# View valid location names
print("Available locations:", le.classes_)

# Pick a known valid location
location_name = "BHMMBMMBX01"  # replace with any one from the above list
encoded_location = le.transform([location_name])[0]

# Example test features
features = [[encoded_location, 13, 4, 0]]  # 6 PM on Saturday, weekend

prediction = model.predict(features)[0]

print(f"\nLocation: {location_name}")
print("Prediction:", "Available ✅" if prediction == 1 else "Not Available ❌")
