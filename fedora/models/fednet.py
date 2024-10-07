from torch import nn
import torch.nn.functional as F


class FedNet(nn.Module):
    def __init__(self, bias = False, num_classes=10):
        super(FedNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, bias=bias)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, bias=bias)
        self.fc1 = nn.Linear(64*5*5, 512, bias=bias)
        self.fc2 = nn.Linear(512, 128, bias=bias)
        self.fc3 = nn.Linear(128, num_classes, bias=bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
