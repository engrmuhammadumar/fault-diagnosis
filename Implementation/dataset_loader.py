import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations for the CWT scalograms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 (adjust based on your model)
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize for better model convergence
])

# Define the path to the dataset
data_dir = r'E:\1 Paper Work\Test CNN Transformer\Data\AE_CWT/'

# Load the dataset from the directory
def get_data_loaders(batch_size=32, shuffle=True):
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # Split into train and test loaders
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return data_loader

# Example usage:
# train_loader = get_data_loaders(batch_size=32)
