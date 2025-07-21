import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- 1. Data Loading & Preprocessing ---

df = pd.read_csv('churn_data.csv')

# Handle non-numeric 'TotalCharges'
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Recommended way to fill NaNs (avoids FutureWarning)
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Prepare data
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
df_processed = pd.get_dummies(df, drop_first=True)

# --- 2. Feature Scaling and Model Training ---

X = df_processed.drop('Churn', axis=1)
y = df_processed['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features to improve model convergence and performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Use the same scaler fitted on train data

# Initialize and train the model on the scaled data
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# --- 3. Results ---

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy on Test Data: {accuracy:.4f}")