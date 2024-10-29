import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('training_data.csv')

# Define the features and target
datapoints = ['acousticness', 'danceability', 'duration', 'energy', 'instrumentalness',
              'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']
X = data[datapoints]
y = data['label']

# Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = data[datapoints + ['label']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation with Label")
plt.show()

# Random Forest for Feature Importance
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_

# Plot feature importances
plt.figure(figsize=(10, 6))
sorted_idx = np.argsort(importances)[::-1]
plt.barh(np.array(datapoints)[sorted_idx], importances[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()

# Recursive Feature Elimination (RFE)
logreg = LogisticRegression(max_iter=10000)
rfe = RFE(logreg, n_features_to_select=6)
rfe.fit(X, y)
rfe_support = rfe.support_
rfe_ranking = rfe.ranking_

# Print features selected by RFE
selected_features = [datapoints[i] for i in range(len(datapoints)) if rfe_support[i]]
print(f"Top 6 Features selected by RFE: {selected_features}")


