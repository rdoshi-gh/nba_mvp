import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load the labeled data
df = pd.read_csv('data/processed/player_stats_labeled.csv')

# Drop irrelevant columns
df = df.drop(columns=['Rk', 'Age', 'Tm', 'Lg', 'GS', 'Season'], errors='ignore')

# Get rid of rows missing values
df = df.dropna()

# Define the features used for training
features = ['PTS', 'AST', 'REB', 'PER', '3P', 'STL', 'BLK', 'FG%', 'FT%']
features = [col for col in features if col in df.columns]

# Select the feature columns (X) and the target column (y)
X = df[features]
y = df['is_mvp']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict the results on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print the classification report (precision, recall, f1-score)
print(classification_report(y_test, y_pred))

os.makedirs('models', exist_ok=True)

# Save the trained model
joblib.dump(model, 'models/random_forest_model.pkl')

print("Model training complete and saved to 'models/random_forest_model.pkl'")
