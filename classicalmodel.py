import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split

# =====================
# 1️⃣ Load Encoded Data
# =====================
def load_encoded_data(one_hot_csv, train_ratio=0.8):
    one_hot_data = pd.read_csv(one_hot_csv, header=None).values.astype(np.float32)
    X_cnn = torch.tensor(one_hot_data, dtype=torch.float32).unsqueeze(1)  # CNN input

    dataset = TensorDataset(X_cnn, X_cnn)  # Autoencoder: input = target
    train_size = max(1, int(train_ratio * len(dataset)))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"Total samples: {len(dataset)}, Train: {train_size}, Test: {test_size}")
    return train_dataset, test_dataset

# =====================
# 2️⃣ Define Classical CNN Autoencoder (No Quantum)
# =====================
class ClassicalEncoder(nn.Module):
    def __init__(self, input_length=7098, latent_dim=5, num_channels=64):
        super(ClassicalEncoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, num_channels, 3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2, stride=2)

        # Compute the correct reduced length dynamically
        temp_input = torch.zeros(1, 1, input_length * 5)  # Simulated input with one-hot encoding
        temp_output = self.pool(torch.relu(self.conv2(torch.relu(self.conv1(temp_input)))))
        actual_reduced_length = temp_output.shape[2]

        self.flattened_size = num_channels * actual_reduced_length
        print(f"Corrected Flattened Size: {self.flattened_size}")

        self.fc = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ClassicalDecoder(nn.Module):
    def __init__(self, output_length=7098, latent_dim=5):
        super(ClassicalDecoder, self).__init__()
        self.output_length = output_length * 5
        self.fc = nn.Linear(latent_dim, (self.output_length // 2) * 64)
        self.deconv1 = nn.ConvTranspose1d(64, 32, 3, stride=1, padding=1)
        self.upsample1 = nn.Upsample(size=35490, mode='nearest')
        self.deconv2 = nn.ConvTranspose1d(32, 1, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 64, self.output_length // 2)
        x = torch.relu(self.deconv1(x))
        x = self.upsample1(x)
        x = torch.sigmoid(self.deconv2(x))
        return x

class ClassicalAutoencoder(nn.Module):
    def __init__(self, input_length=7098, latent_dim=5):
        super(ClassicalAutoencoder, self).__init__()
        self.encoder = ClassicalEncoder(input_length, latent_dim)
        self.decoder = ClassicalDecoder(input_length, latent_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

# =====================
# 3️⃣ Train and Evaluate the Model
# =====================
import time

def train_model(model, train_dataset, epochs=10, batch_size=16, learning_rate=0.001):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    start_time = time.time()  # Start time tracking

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for X_cnn, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(X_cnn)
            loss = criterion(outputs, X_cnn)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.6f}")

    end_time = time.time()  # End time tracking
    elapsed_time = end_time - start_time
    print(f"Total Training Time: {elapsed_time:.2f} seconds")

def evaluate_model(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    start_time = time.time()  # Start time tracking

    with torch.no_grad():
        for X_cnn, _ in test_loader:
            outputs = model(X_cnn)
            loss = criterion(outputs, X_cnn)
            total_loss += loss.item()

    end_time = time.time()  # End time tracking
    elapsed_time = end_time - start_time

    print(f"Test Loss: {total_loss:.6f}")
    print(f"Evaluation Time: {elapsed_time:.2f} seconds")


# =====================
# 4️⃣ Run Training (Classical Model)
# =====================
one_hot_csv = "one_hot_encoded.csv"

train_dataset, test_dataset = load_encoded_data(one_hot_csv)
model = ClassicalAutoencoder()
train_model(model, train_dataset, epochs=10)
evaluate_model(model, test_dataset)
