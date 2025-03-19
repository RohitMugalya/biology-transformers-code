import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import pennylane as qml
from torch.utils.data import DataLoader, TensorDataset, random_split

# =====================
# 1️⃣ Load Encoded Data
# =====================
def load_encoded_data(one_hot_csv, quantum_csv, train_ratio=0.8):
    one_hot_data = pd.read_csv(one_hot_csv, header=None).values.astype(np.float32)
    quantum_data = pd.read_csv(quantum_csv, header=None).values.astype(np.float32)

    X_cnn = torch.tensor(one_hot_data, dtype=torch.float32).unsqueeze(1)  # CNN input
    X_qcnn = torch.tensor(quantum_data, dtype=torch.float32)  # QCNN input

    dataset = TensorDataset(X_cnn, X_qcnn)
    train_size = max(1, int(train_ratio * len(dataset)))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"Total samples: {len(dataset)}, Train: {train_size}, Test: {test_size}")
    return train_dataset, test_dataset
# =====================
# 2️⃣ Define Hybrid CNN + QCNN Model
# =====================
class ClassicalEncoder(nn.Module):
    def __init__(self, input_length=7098, latent_dim=20, num_channels=64):  # 🔹 Increased latent_dim
        super(ClassicalEncoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, num_channels, 3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2, stride=2)

        # Compute reduced length dynamically (only one pooling)
        temp_input = torch.zeros(1, 1, input_length * 5)
        temp_output = self.pool(torch.relu(self.conv2(torch.relu(self.conv1(temp_input)))))
        actual_reduced_length = temp_output.shape[2]  # Get correct sequence length after pooling

        self.flattened_size = num_channels * actual_reduced_length
        print(f"Updated Flattened Size: {self.flattened_size}")

        self.fc = nn.Linear(self.flattened_size, latent_dim)  # 🔹 Increased latent_dim

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))  
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# =====================
# 3️⃣ Quantum Latent Compressor (PennyLane QCNN)
# =====================
class QuantumLatentCompressor(nn.Module):
    def __init__(self, n_qubits=20):  # 🔹 Increased quantum encoding size
        super(QuantumLatentCompressor, self).__init__()
        self.n_qubits = n_qubits
        self.params = nn.Parameter(torch.randn(n_qubits))

        self.device = qml.device("lightning.qubit", wires=n_qubits)

        @qml.qnode(self.device, interface="torch")
        def quantum_circuit(inputs):
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.quantum_circuit = quantum_circuit

    def forward(self, x):
        quantum_output = self.quantum_circuit(self.params)  
        quantum_output = torch.tensor(quantum_output, dtype=torch.float32).expand(x.shape[0], -1)
        return quantum_output

# =====================
# 4️⃣ Classical Decoder
# =====================
class ClassicalDecoder(nn.Module):
    def __init__(self, output_length=7098, latent_dim=20):  # 🔹 Increased latent_dim
        super(ClassicalDecoder, self).__init__()
        self.output_length = output_length * 5  
        self.fc = nn.Linear(latent_dim, (self.output_length // 2) * 64)  # 🔹 Adjusted for less compression
        self.deconv1 = nn.ConvTranspose1d(64, 32, 3, stride=1, padding=1)
        self.upsample1 = nn.Upsample(size=35490, mode='nearest')  # 🔹 Upsample directly
        self.deconv2 = nn.ConvTranspose1d(32, 1, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 64, self.output_length // 2)  
        x = torch.relu(self.deconv1(x))
        x = self.upsample1(x)  # 🔹 Upsample to match original size
        x = torch.sigmoid(self.deconv2(x))
        return x


# =====================
# 5️⃣ Full Hybrid Model
# =====================
class HybridGenCoder(nn.Module):
    def __init__(self, input_length=7098, latent_dim=5):
        super(HybridGenCoder, self).__init__()
        self.encoder = ClassicalEncoder(input_length, latent_dim)
        self.quantum_compressor = QuantumLatentCompressor(n_qubits=latent_dim)
        self.decoder = ClassicalDecoder(input_length, latent_dim)

    def forward(self, x):
        encoded = self.encoder(x)  # (batch, 5)
        quantum_compressed = self.quantum_compressor(encoded)  # (batch, 5)
        reconstructed = self.decoder(quantum_compressed)  # (batch, 1, 35490)
        return reconstructed

# =====================
# 6️⃣ Train and Evaluate the Model
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
# 7️⃣ Run Training
# =====================
one_hot_csv = "one_hot_encoded.csv"
quantum_csv = "quantum_encoded.csv"

train_dataset, test_dataset = load_encoded_data(one_hot_csv, quantum_csv)
model = HybridGenCoder()
train_model(model, train_dataset, epochs=10)
evaluate_model(model, test_dataset)
