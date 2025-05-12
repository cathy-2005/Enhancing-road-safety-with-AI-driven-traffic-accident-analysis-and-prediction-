import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load your dataset
df = pd.read_csv(r"C:\Users\Administrator\Desktop\New folder\accident_prediction_india.csv")

# Create a directory to store encoders
encoder_dir = "encoders"
os.makedirs(encoder_dir, exist_ok=True)

# Define categorical columns to encode (edit as needed)
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Function to safely transform using LabelEncoder
def safe_transform(encoder, data):
    transformed = []
    for item in data:
        if item in encoder.classes_:
            transformed.append(encoder.transform([item])[0])
        else:
            transformed.append(-1)  # Unseen label
    return np.array(transformed)

# Apply Label Encoding and save each encoder
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    le.fit(df[col])
    
    # Save encoder to file
    with open(os.path.join(encoder_dir, f"{col}_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    # Replace column with encoded values
    df[col] = safe_transform(le, df[col])

# Preview encoded data
print(df.head())

# Optional: Save encoded data
df.to_csv("encoded_accident_data.csv", index=False)

