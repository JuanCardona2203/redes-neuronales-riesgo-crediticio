import torch
import torch.nn as nn

class CreditRiskMLP(nn.Module):
    def __init__(self, input_dim):
        super(CreditRiskMLP, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 16)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.hidden1(x)
        out = self.relu1(out)
        out = self.hidden2(out)
        out = self.relu2(out)
        out = self.output(out)
        out = self.sigmoid(out)
        return out
