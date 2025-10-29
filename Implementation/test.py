import torch
from dataset_loader import get_data_loaders
from model import MCTFaultDetectionModel

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the saved model
model = MCTFaultDetectionModel().to(device)
model.load_state_dict(torch.load('mct_fault_detection_model.pth'))
model.eval()

# Load the test set
test_loader = get_data_loaders(batch_size=32)

# Testing function
def test_model():
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy}%')

if __name__ == "__main__":
    test_model()
