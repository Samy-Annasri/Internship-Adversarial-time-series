import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, features) need to change to
        # PyTorch Conv1d: (batch_size, features, sequence_length)
        x = x.permute(0, 2, 1)  # Reorder to (batch_size, features, sequence_length)

        x = self.relu1(self.conv1(x))  # Apply first conv + ReLU
        x = self.relu2(self.conv2(x))  # Apply second conv + ReLU
        x = self.pool(x)               # Shape becomes (batch_size, 64, 1)

        x = x.view(x.size(0), -1)      # Flatten to (batch_size, 64)
        return self.fc(x)              # Final output: (batch_size, output_size)