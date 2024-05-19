import torch.nn as nn
import torch.nn.functional as F


class MFCC_DNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MFCC_DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 144)
        self.fc2 = nn.Linear(144, 144)
        self.fc3 = nn.Linear(144, 144)
        self.classifier = nn.Linear(144, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.classifier(x)
        return x
