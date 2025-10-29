import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class CNNBlock(nn.Module):
    def __init__(self):
        super(CNNBlock, self).__init__()
        # Use pretrained ResNet18 as the backbone CNN
        self.cnn = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.cnn.fc = nn.Identity()  # Remove the classification layer

    def forward(self, x):
        return self.cnn(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super(TransformerBlock, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True), 
            num_layers=num_layers
        )

    def forward(self, x):
        x = self.transformer(x)
        x = x.mean(dim=1)  # Average over the sequence dimension
        return x

class MCTFaultDetectionModel(nn.Module):
    def __init__(self):
        super(MCTFaultDetectionModel, self).__init__()
        self.cnn_block = CNNBlock()  # CNN (ResNet18 backbone)
        self.transformer_block = TransformerBlock()

        # Output for multi-class classification (4 classes: BF, GF, TF, N)
        self.classification_head = nn.Linear(512, 7)

    def forward(self, x):
        cnn_features = self.cnn_block(x)  # Extract features using ResNet
        cnn_features = cnn_features.unsqueeze(1)  # Add sequence dimension for Transformer
        transformer_features = self.transformer_block(cnn_features)  # Apply Transformer

        # Multi-class classification (4 classes)
        output = self.classification_head(transformer_features)
        return output
