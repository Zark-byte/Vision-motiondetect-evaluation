import torch # PyTorch developed by Meta
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json


with open("X_Lable.json",'r') as load_f:
    X_Lable = json.load(load_f)
with open("Y_Lable.json",'r') as load_f:
    Y_Lable = json.load(load_f)
with open("Lable.json",'r') as load_f:
    test = json.load(load_f)

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, hidden = self.rnn(x)
        out = self.fc(hidden.squeeze(0))
        return out


class TimeSeriesDataset(Dataset):
    def __init__(self):
        self.data = []
        self.labels = []
        for i in range(8):
            self.data.append(np.array(X_Lable[i]))
            self.labels.append(Y_Lable[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(X_Lable[idx]), torch.LongTensor([Y_Lable[idx]])

# Hyperparameters
input_size = 2
hidden_size = 64
output_size = 3
batch_size = 1
num_epochs = 20
learning_rate = 0.005

# Create dataset and dataloader
dataset = TimeSeriesDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Initialize the model, loss function, and optimizer
model = RNN(input_size, hidden_size, output_size)
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
        loss = criterion(outputs, labels.flatten())

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
with torch.no_grad():
    test = torch.tensor(test)
    test_reshaped = test.view(test.size(0), test.size(1), 1)
    test_output = model(test_reshaped)
    predicted_class = torch.argmax(test_output).item()
print(f"Test input classification: {predicted_class+1}")
#OUTPUT show
if predicted_class==0:
    print("Your posture reaches low level. You should practice more and pay attention to your motion")
if predicted_class==1:
    print("Your posture reaches mediate level. Good job! You could improve yourself by finding a frofessional reacher.")
if predicted_class==2:
    print("Your posture reaches high level. Congratulations!")
