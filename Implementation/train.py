import torch
import torch.nn as nn
import torch.optim as optim
from dataset_loader import get_data_loaders
from model import MCTFaultDetectionModel

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 32
learning_rate = 1e-3
epochs = 10

# Load DataLoader for training
train_loader = get_data_loaders(batch_size=batch_size)

# Initialize model, loss function, and optimizer
model = MCTFaultDetectionModel().to(device)
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train():
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}, Accuracy: {accuracy}%')

if __name__ == "__main__":
    train()
