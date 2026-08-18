import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("heart.csv")

# --- PLACE THE MAPPING CODE HERE ---
# Convert categorical 'thal' column to numeric so scikit-learn can use it
thal_map = {'normal': 0, 'fixed': 1, 'reversible': 2}
df['thal'] = df['thal'].map(thal_map).fillna(3).astype(int)
# -----------------------------------

# 2. Print columns (These lines can stay exactly as they were)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head())

# 3. Separate features and target (The rest of your code continues below)
X = df.drop("target", axis=1)
y = df["target"]
# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2%}")

# Test a single prediction
sample = X_test.iloc[[0]]
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0]
print(f"\nSample prediction: {prediction}")
print(f"Probability array: {probability}")
print("Notice: these raw numbers mean nothing without context!")