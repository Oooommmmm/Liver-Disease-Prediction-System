# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split  # Added for splitting
from sklearn.metrics import accuracy_score            # Added for evaluation
import pickle

# 1. Load data (Make sure indian_liver_patient.csv is in the same folder)
df = pd.read_csv('indian_liver_patient.csv')

# 2. Preprocess 
# Fill missing values in Albumin/Globulin ratio
df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean(), inplace=True)

# Encode Gender (Male = 1, Female = 0)
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Format Target Variable (Disease = 1, No Disease = 0)
df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})

# 3. Split Features (X) and Target (y)
X = df.drop(columns=['Dataset'])
y = df['Dataset']

# NEW: Split data into Training set (80%) and Testing set (20%)
# random_state=42 ensures the split is reproducible every time you run it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
# Using Random Forest as decided in your MLBI SLA document
# We now fit the model ONLY on the training data (X_train, y_train)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# NEW: Evaluate the model on unseen test data
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# 5. Save the trained model to a file
with open('liver_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained successfully!")
print(f"Calculated Test Accuracy: {accuracy * 100:.2f}%")
print("Saved as 'liver_model.pkl'. You can now run app.py")