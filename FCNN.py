import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import copy

# Load Data
data = pd.read_csv('training_data.csv', sep=',')
test_data = pd.read_csv('songs_to_classify.csv', sep=',')

X = data[['acousticness', 'danceability', 'energy', 'liveness', 'speechiness', 'instrumentalness']]
y = data['label'].values
songs = test_data[['acousticness', 'danceability', 'energy', 'liveness', 'speechiness', 'instrumentalness']]

X_combined = pd.concat([X, songs])

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_combined)

# Split data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled[:len(X)], y, test_size=0.2, random_state=42)
songs = X_scaled[len(X):]

# Convert data into PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
songs_tensor = torch.tensor(songs, dtype=torch.float32)


# Define the Neural Network
class SongClassifier(nn.Module):
    def __init__(self):
        super(SongClassifier, self).__init__()
        self.fc1 = nn.Linear(X_train_tensor.shape[1], 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


model = SongClassifier()

def model_train(model, X_train, y_train, X_val, y_val, songs):
    loss_history = []
    accuracy_history_tr = []
    accuracy_history_val = []
    loss_fn = nn.BCELoss() # For our Binary classification task
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 700 # number of epochs to run
    song_labels = []

    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None

    for epoch in range(n_epochs):
        model.train()
        # forward pass
        y_pred = model(X_train)
        training_loss = loss_fn(y_pred, y_train)
        loss_history.append(training_loss.item())
        # backward pass
        optimizer.zero_grad()
        training_loss.backward()
        # update weights
        optimizer.step()

        training_acc = (y_pred.round() == y_train).float().mean()
        accuracy_history_tr.append(training_acc)

        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        val_acc = (y_pred.round() == y_val).float().mean()
        accuracy_history_val.append(val_acc)
        test_acc = float(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

        # print progress
        print(f"Epoch [{epoch}], Training Loss: {training_loss:.4f}, Accuracy: {training_acc:.4f}"
              f", Validation acc: {test_acc:.4f}")

    # Plot the loss and accuracy over epochs
    fig = plt.figure(figsize=(12, 10))

    gs = GridSpec(2, 2, height_ratios=[0.5, 0.5])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(n_epochs), accuracy_history_val, label='Testing Accuracy', color='red')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Validation Accuracy')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(n_epochs), accuracy_history_tr, label='Training Accuracy', color='green')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')

    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(range(n_epochs), loss_history, label='Training Loss', color='blue')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Loss')

    plt.tight_layout()

    plt.show()

    # restore model and return best accuracy
    model.load_state_dict(best_weights)

    # Predict song labels with restored model
    model.eval()
    for song in songs:
        y_song = model(song).squeeze()
        song_labels.append(round(y_song.item()))
    return best_acc, song_labels

acc, song_labels = model_train(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, songs_tensor)

print("Best Accuracy: %.4f" % acc)
print(*song_labels, sep='')

