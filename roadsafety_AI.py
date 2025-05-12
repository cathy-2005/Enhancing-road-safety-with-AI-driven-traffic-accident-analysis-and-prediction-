import pandas as pd

# Load the dataset
df = pd.read_csv(r"C:\Users\Administrator\Desktop\New folder\accident_prediction_india.csv")  # Replace with the correct file path

# Clean the column names (strip spaces)
df.columns = df.columns.str.strip()

# Check and drop 'Accident Severity' if it exists
if 'Accident Severity' in df.columns:
    df = df.drop('Accident Severity', axis=1)
    print("'Accident Severity' column dropped successfully.")
else:
    print("'Accident Severity' column not found in the dataset.")

# Print the first few rows of the dataset to verify
print("Dataset preview:\n", df.head())
