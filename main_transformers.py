import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from transformers import EsmTokenizer, EsmModel
from torch.utils.data import DataLoader, TensorDataset, random_split

# =====================
# 1️⃣ Load Encoded Data
# =====================
def load_encoded_data(one_hot_csv, train_ratio=0.8):
    one_hot_data = pd.read_csv(one_hot_csv, header=None).values.astype(str).flatten().tolist()
    dataset = TensorDataset(torch.tensor(range(len(one_hot_data))))
    train_size = max(1, int(train_ratio * len(dataset)))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"Total samples: {len(dataset)}, Train: {train_size}, Test: {test_size}")
    return one_hot_data, train_dataset, test_dataset

# =====================
# Utility: One-Hot Encode a Nucleotide Sequence
# =====================
def one_hot_encode(seq, alphabet="ACGTN", seq_length=7098):
    """
    Convert a nucleotide sequence (string) into a one-hot encoded tensor.
    Ensures output shape: (1, seq_length * num_classes), where num_classes=5 (A,C,G,T,N).
    """
    one_hot = torch.zeros(seq_length, len(alphabet), dtype=torch.float32)
    for i, char in enumerate(seq[:seq_length]):  # Truncate if longer
        if char in alphabet:
            one_hot[i, alphabet.index(char)] = 1.0
    return one_hot.flatten().unsqueeze(0)  # shape: (1, seq_length * 5)

# =====================
# 2️⃣ Define Nucleotide Transformer Encoder
# =====================
class NucleotideTransformerEncoder(nn.Module):
    def __init__(self, model_name="InstaDeepAI/nucleotide-transformer-v2-50m-3mer-multi-species"):
        super(NucleotideTransformerEncoder, self).__init__()
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name, ignore_mismatched_sizes=True)
        self.latent_dim = self.model.config.hidden_size

    def forward(self, sequences):
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(next(self.model.parameters()).device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

# =====================
# 3️⃣ Classical Decoder
# =====================
class ClassicalDecoder(nn.Module):
    def __init__(self, output_length=7098, latent_dim=512):  
        super(ClassicalDecoder, self).__init__()
        self.output_length = output_length * 5  
        self.fc = nn.Linear(latent_dim, (self.output_length // 4) * 64)
        self.deconv1 = nn.ConvTranspose1d(64, 32, 3, stride=1, padding=1)
        self.upsample1 = nn.Upsample(size=35490 // 2, mode='nearest')
        self.deconv2 = nn.ConvTranspose1d(32, 16, 3, stride=1, padding=1)
        self.upsample2 = nn.Upsample(size=35490, mode='nearest')
        self.deconv3 = nn.ConvTranspose1d(16, 1, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 64, self.output_length // 4)
        x = torch.relu(self.deconv1(x))
        x = self.upsample1(x)
        x = torch.relu(self.deconv2(x))
        x = self.upsample2(x)
        x = torch.sigmoid(self.deconv3(x))
        return x  # Output shape: (batch, 1, 35490)

# =====================
# 4️⃣ Full Hybrid Model
# =====================
class HybridGenCoder(nn.Module):
    def __init__(self, model_name):
        super(HybridGenCoder, self).__init__()
        self.encoder = NucleotideTransformerEncoder(model_name)
        self.decoder = ClassicalDecoder(latent_dim=self.encoder.latent_dim)

    def forward(self, sequences):
        encoded = self.encoder(sequences)  
        reconstructed = self.decoder(encoded)  
        return reconstructed  # Shape: (batch, 1, 35490)

# =====================
# 5️⃣ Train and Evaluate the Model
# =====================
def train_model(model, one_hot_data, train_dataset, epochs=10, batch_size=16, learning_rate=0.001, device='cpu'):
    model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_indices in train_loader:
            batch_sequences = [one_hot_data[idx] for idx in batch_indices[0].tolist()]
            targets = [one_hot_encode(seq).to(device) for seq in batch_sequences]  
            targets = torch.cat(targets, dim=0).unsqueeze(1)  # Shape: (batch, 1, 35490)

            optimizer.zero_grad()
            outputs = model(batch_sequences)  # Shape: (batch, 1, 35490)

            loss = criterion(outputs, targets)  
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.6f}")

def evaluate_model(model, one_hot_data, test_dataset, device='cpu'):
    model.to(device)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch_indices in test_loader:
            batch_sequences = [one_hot_data[idx] for idx in batch_indices[0].tolist()]
            targets = [one_hot_encode(seq).to(device) for seq in batch_sequences]
            targets = torch.cat(targets, dim=0).unsqueeze(1)  # Shape: (batch, 1, 35490)
            outputs = model(batch_sequences)  
            loss = criterion(outputs, targets)  
            total_loss += loss.item()
    print(f"Test Loss: {total_loss:.6f}")

# =====================
# 6️⃣ Run Training
# =====================
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    one_hot_csv = "one_hot_encoded.csv"
    one_hot_data, train_dataset, test_dataset = load_encoded_data(one_hot_csv)
    model = HybridGenCoder("InstaDeepAI/nucleotide-transformer-v2-50m-3mer-multi-species")
    train_model(model, one_hot_data, train_dataset, epochs=10, device=device)
    evaluate_model(model, one_hot_data, test_dataset, device=device)
