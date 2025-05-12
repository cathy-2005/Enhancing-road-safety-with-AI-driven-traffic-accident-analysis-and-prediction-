import pandas as pd

# Load dataset
print("Loading dataset...")
df = pd.read_csv(r"C:\Users\Administrator\Desktop\New folder\accident_prediction_india.csv")  # Update the path

# Show dataset preview
print("Dataset preview:")
print(df.head())

# Example preprocessing steps
df.columns = df.columns.str.strip()  # Clean column names

# Check and drop columns if they exist
columns_to_drop = ['Accident_Severity', 'Light Conditions']
existing_columns = [col for col in columns_to_drop if col in df.columns]
print(f"Dropping columns: {existing_columns}")
df = df.drop(existing_columns, axis=1)

# Final dataset preview
print("Dataset after preprocessing:")
print(df.head())

# Save the processed data to a new file
df.to_csv("processed_data.csv", index=False)
print("Data saved as 'processed_data.csv'")

