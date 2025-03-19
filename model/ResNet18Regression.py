import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Regression(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet18Regression, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        
        # Modify the first conv layer to accept single-channel images (if required)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the final fully-connected layer to output single regression value
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.model(x)
