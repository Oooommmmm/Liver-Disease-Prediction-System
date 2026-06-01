import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score            
import pickle

df = pd.read_csv('indian_liver_patient.csv')


df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean(), inplace=True)


df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})


df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})


X = df.drop(columns=['Dataset'])
y = df['Dataset']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

with open('liver_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained successfully!")
print(f"Calculated Test Accuracy: {accuracy * 100:.2f}%")
print("Saved as 'liver_model.pkl'. You can now run app.py")
