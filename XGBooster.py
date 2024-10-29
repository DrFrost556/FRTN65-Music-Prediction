import xgboost as xgb
from numpy.f2py.cfuncs import callbacks
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Data
data = pd.read_csv('training_data.csv')
test_data = pd.read_csv('songs_to_classify.csv')

test_data = test_data[['acousticness', 'danceability', 'energy', 'liveness', 'speechiness', 'instrumentalness']]


# Prepare and Standardize Data
X = data[['acousticness', 'danceability', 'energy', 'liveness', 'speechiness', 'instrumentalness']]
y = data['label'].values


# Concatenate with test data for scaling
X_combined = pd.concat([X, test_data])
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_combined)

# Split into training, validation, and test sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled[:len(X)], y, test_size=0.2, random_state=42)
songs = X_scaled[len(X):]

# Hyperparameter Optimization using GridSearchCV, reduced to already found best parameters to decrease training time.
params = {
    'objective': ['multi:softmax'],
    'num_class': [2],
    'max_depth': [16],
    'eta': [0.3],
    'subsample': [1],
    'colsample_bytree': [0.6],
    'gamma': [0.5],  # Adding regularization hyperparameters
    'min_child_weight': [1],  # Controls complexity
    'lambda': [1],  # L2 regularization
    'alpha': [1],  # L1 regularization
    'max_delta_step': [1],  # Useful for imbalanced data
}

# Initialize the XGBClassifier
xgb_model = xgb.XGBClassifier(verbosity=0, eval_metric='merror')

# Perform Grid Search
grid_search = GridSearchCV(estimator=xgb_model, param_grid=params, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Get best parameters and evaluate on validation set
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# Updated parameters for xgb.train
train_params = {
    'objective': 'multi:softmax',
    'num_class': 2,
    'max_depth': best_params['max_depth'],
    'eta': best_params['eta'],
    'subsample': best_params['subsample'],
    'colsample_bytree': best_params['colsample_bytree'],
    'gamma': best_params['gamma'],
    'eval_metric': 'merror',
    'min_child_weight': best_params['min_child_weight'],  # Controls complexity
    'lambda': best_params['lambda'],  # L2 regularization
    'alpha': best_params['alpha'],  # L1 regularization
    'max_delta_step': best_params['max_delta_step'],  # Useful for imbalanced data
    'verbosity': 0,
}

# Track evaluation metrics for visualization
eval_result = {}

# Train the model using xgb.train with early stopping
final_model = xgb.train(
    train_params,
    dtrain,
    num_boost_round=200,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=10,
    evals_result=eval_result,
    verbose_eval=True
)

train_error = eval_result['train']['merror']
val_error = eval_result['val']['merror']


plt.figure(figsize=(10, 6))
plt.plot(train_error, label='Train Error')
plt.plot(val_error, label='Validation Error')
plt.title("Training and Validation Error over Boosting Rounds")
plt.xlabel("Boosting Rounds")
plt.ylabel("Error Rate")
plt.legend()
plt.show()

# Predict on validation data
y_val_pred = final_model.predict(dval)
y_val_pred = np.round(y_val_pred)

# Calculate accuracy on the validation set
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Predict on test data
dtest = xgb.DMatrix(songs)
y_pred = final_model.predict(dtest)
y_pred = np.round(y_pred)

# Save Model
final_model.save_model('xgboost_model.json')

# Output Predictions
predictions = [round(value) for value in y_pred]
print(*predictions, sep='')
