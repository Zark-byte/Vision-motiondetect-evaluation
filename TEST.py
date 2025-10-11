import torch # PyTorch developed by Meta
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Define the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, hidden = self.rnn(x)
        out = self.fc(hidden.squeeze(0))
        return out

# Create a sample dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, num_samples, seq_length):
        self.data = []
        self.labels = []
        for _ in range(num_samples):
            # Generate random time series data
            seq = np.random.randn(seq_length)
            label = 1 if np.sum(seq) > 0 else 0  # Simple classification rule
            self.data.append(seq)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])

# Hyperparameters
input_size = 1
hidden_size = 64
output_size = 2  # Binary classification
seq_length = 20
num_samples = 1000
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Create dataset and dataloader
dataset = TimeSeriesDataset(num_samples, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
model = SimpleRNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.unsqueeze(2)  # Add input_size dimension

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels.squeeze())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.squeeze()).sum().item()

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

print("Training finished!")

# Test the model
test_input = torch.randn(1, seq_length, 1)  # Create a random test input
with torch.no_grad():
    test_output = model(test_input)
    predicted_class = torch.argmax(test_output).item()
print(f"Test input classification: {predicted_class}")