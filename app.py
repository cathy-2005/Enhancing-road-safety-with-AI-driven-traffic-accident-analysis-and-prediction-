import joblib
import pandas as pd

# Load the trained model
model = joblib.load("road_safety_model.pkl")
print("Model loaded successfully.")

# Load the column names from the training data
training_data = pd.read_csv("processed_data.csv")
X_columns = training_data.drop("Accident Severity", axis=1).columns

# Create new input with **all required columns**
# Replace these example values with actual inputs as needed
new_data = pd.DataFrame([{
    col: 'Unknown' for col in X_columns  # Default dummy values
}])

# Customize a few inputs manually
new_data['State Name'] = 'Tamil Nadu'
new_data['City Name'] = 'Chennai'
new_data['Alcohol Involvement'] = 'Yes'
new_data['Accident Location Details'] = 'Intersection'
# Set other required values here...

# Encode using same logic used in training
from sklearn.preprocessing import LabelEncoder

for column in new_data.columns:
    new_data[column] = LabelEncoder().fit_transform(new_data[column].astype(str))

# Predict
prediction = model.predict(new_data)
print("Predicted Accident Severity:", prediction[0])


