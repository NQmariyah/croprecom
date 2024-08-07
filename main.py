import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib


# Load the CSV file with the correct delimiter
file_path = 'Crop_recommendation.csv'
data = pd.read_csv(file_path, delimiter=';')

# Encode the labels
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])
print("Encode data:\n", data.head())

# Separate features and labels
X = data.drop('label', axis=1)
y = data['label']
print("\nFeatures (X):\n", X.head())
print("\nLabels (y):\n", y.head())

# Normalize the features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
print("\nNormalize Data:\n", pd.DataFrame(X_normalized).head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Build the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model on the test set
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(model, 'crop_prediction_model.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')
joblib.dump(scaler, 'scaler.joblib')
