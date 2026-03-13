import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("autism_screening.csv")

#show column names
print(data.columns)

# Show first 5 rows
print(data.head())

# Select features and target
X = data.select_dtypes(include=['int64','float64'])
y = data["Class/ASD"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Create a sample using correct number of features
sample = X.iloc[0:1]

prediction = model.predict(sample)

print("Prediction:", prediction)