import torch
import torch.nn as nn

class SimpleLSTM(torch.nn.Module):
  def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.2):
    super(SimpleLSTM, self).__init__()
    self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers,batch_first=True, dropout=dropout)
    self.linear = torch.nn.Linear(hidden_size, output_size)

  def forward(self, x):
    x, _ = self.lstm(x)
    x = self.linear(x[:, -1, :])
    return x