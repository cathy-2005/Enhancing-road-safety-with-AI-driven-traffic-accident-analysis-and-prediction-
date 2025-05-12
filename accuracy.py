import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# ✅ Load dataset
df = pd.read_csv(r"C:\Users\Administrator\Desktop\New folder\accident_prediction_india.csv")

# ✅ Set your target column here
target_column = 'Accident Severity'  # Change this if needed

# ✅ Drop rows where target is missing
df = df.dropna(subset=[target_column])

# ✅ Encode categorical columns
le_dict = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

# ✅ Define features (X) and target (y)
X = df.drop(columns=[target_column])
y = df[target_column]

# ✅ Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Check if target is classification or regression
if len(y.unique()) <= 10 and y.dtype in [np.int64, np.int32]:
    print("🎯 Detected classification task.")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("✅ Accuracy:", acc)
else:
    print("📊 Detected regression task.")
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print("✅ RMSE:", rmse)
    print("✅ R² Score:", r2)

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)

from sklearn.metrics import mean_squared_error
import numpy as np

# For regression tasks, we calculate RSME
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RSME):", rmse)

