import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier  # 🔄 Changed from LogisticRegression to SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib
import os
 
# Load dataset
df = pd.read_csv(r"C:\Users\SanjaySiramdasu\Downloads\Technical Val Exceptions(Scenarios (2)).csv", encoding='latin1')
 
# Preprocess the data
df['Error Message'] = df['Error Message'].astype(str).str.lower().str.strip()
df['Expected Output'] = df['Expected Output'].astype(str).str.lower().str.strip()
 
# Split data: first 21 records for training, remaining for testing
X_train = df['Error Message'][:21]
y_train = df['Expected Output'][:21]
X_test = df['Error Message'][21:]
y_test = df['Expected Output'][21:]
 
# Check training/testing data
print("\n🔹 Training data:")
print(X_train)
print("\n🔹 Testing data:")
print(X_test)
print("\n🔹 Training labels:")
print(y_train)
print("\n🔹 Testing labels:")
print(y_test)
 
# Create a folder to save the model
os.makedirs('model', exist_ok=True)
 
# Build ML pipeline with SGDClassifier
model = make_pipeline(
    TfidfVectorizer(),
    SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
)
 
# Train the model
model.fit(X_train, y_train)
 
# Save the model to a file
joblib.dump(model, 'model/sgd_classifier_model.pkl')
 
# Predict using the model on test data
y_pred = model.predict(X_test)
 
# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
 
print("\n✅ SGD Classifier model trained and saved successfully.")
print("🔹 Accuracy on the test set:", accuracy)