import torch
import torch.nn as nn
import torchvision.models as models


class EmotionResNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # Load pretrained ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 🔧 Convert RGB → Grayscale
        weight = self.model.conv1.weight.data
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1.weight.data = weight.mean(dim=1, keepdim=True)

        # 🔧 Replace final layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)