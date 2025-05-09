import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from ftfy import fix_text



# Load the trained model
model = joblib.load('models/random_forest_model.pkl')

df = pd.read_csv('data/processed/player_stats_labeled.csv', encoding='utf-8')
df['Player'] = df['Player'].apply(fix_text)

#Correct season
df = df[df['Season'] == '2024-25']

# Filter out low games played
df = df[df['G'] >= 60]

features = ['PTS', 'AST', 'REB', 'PER', '3P', 'STL', 'BLK', 'FG%', 'FT%']
features = [col for col in features if col in df.columns]
X = df[features]

# Make predictions
predictions = model.predict(X)

# Add the prediction to the DataFrame
df['predicted_mvp'] = predictions

probabilities = model.predict_proba(X)[:, 1]
total = probabilities.sum()
df['mvp_probability'] = probabilities / total


# Show top 10 candidates
top_candidates = df.sort_values(by='mvp_probability', ascending=False).head(10)
print(top_candidates[['Player', 'PTS', 'AST', 'mvp_probability']])
df.to_csv('data/processed/predicted_mvp_2025.csv', index=False)
