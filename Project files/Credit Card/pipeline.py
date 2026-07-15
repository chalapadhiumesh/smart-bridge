import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Datasets (Assuming standard Kaggle/UCI Credit Card dataset layout)
print("Loading data...")
app_df = pd.read_csv('dataset/application_record.csv')
credit_df = pd.read_csv('dataset/credit_record.csv')

# 2. Feature Engineering on Credit Record (Target Generation)
# Status definitions: 0: 1-29 days past due, 1: 30-59 days, 2: 60-89 days...
# We define "Bad" (1) as anyone with debts past due for 30+ days (Status 1-5)
credit_df['target'] = credit_df['STATUS'].isin(['1', '2', '3', '4', '5']).astype(int)

# Group by ID to see if an applicant was ever classified as a high-risk ("Bad") customer
target_df = credit_df.groupby('ID')['target'].max().reset_index()

# Merge back with Application Records
df = pd.merge(app_df, target_df, on='ID', how='inner')

# 3. Data Cleaning & Pre-processing
df.drop(['ID'], axis=1, inplace=True)
df.dropna(inplace=True)  # Drops rows with missing values (e.g., OCCUPATION_TYPE)

# Encode Categorical Variables
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate Features and Target
X = df.drop(['target'], axis=1)
y = df['target']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 5. Model Building (Random Forest Classifier)
print("Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Save Artifacts for Flask Application
with open('model.pkl', 'wb') as m_file:
    pickle.dump(model, m_file)

with open('scaler.pkl', 'wb') as s_file:
    pickle.dump(scaler, s_file)

with open('encoders.pkl', 'wb') as e_file:
    pickle.dump(label_encoders, e_file)

print("All pipeline artifacts saved successfully!")