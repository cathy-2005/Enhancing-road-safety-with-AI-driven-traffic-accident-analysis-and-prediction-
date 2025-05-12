import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib

# Load data
df = pd.read_csv(r"C:\Users\Administrator\Desktop\New folder\accident_prediction_india.csv")

# Encode categorical features
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column].astype(str))

# Separate features and target
X = df.drop("Accident Severity", axis=1)
y = df["Accident Severity"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Save model
joblib.dump(model, "road_safety_model.pkl")
print("Model saved as 'road_safety_model.pkl'")
