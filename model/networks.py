import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, layers, device):
        super(MLP, self).__init__()
        self.layers = nn.Sequential().to(device)

        for i in range(len(layers)-2):
            linear = nn.Linear(layers[i], layers[i+1]).to(device)
            batchnorm = nn.BatchNorm1d(layers[i+1]).to(device)
            relu = nn.ReLU(inplace=True).to(device)
            nn.init.kaiming_uniform_(linear.weight, a=0, mode='fan_in', nonlinearity='relu')

            self.layers.add_module("Linear%d" % (i), linear)
            self.layers.add_module("BatchNorm%d" % (i), batchnorm)
            self.layers.add_module("ReLU%d" % (i), relu)

        linear = nn.Linear(layers[-2], layers[-1]).to(device)
        nn.init.kaiming_uniform_(linear.weight, a=0, mode='fan_in', nonlinearity='relu')
        self.layers.add_module("Linear%d" % (len(layers)-2), linear)
        self.device = device


    def forward(self, x):
        x = self.layers(x)
        nan_mask = torch.isnan(x)
        x[nan_mask] = 0
        return x


class CNN(nn.Module):
    def __init__(self, input, output, device):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input, 4, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            # nn.Conv1d(4, 8, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=3, stride=2)
        ).to(device)

        self.fc_layers = nn.Sequential(
            nn.Linear(4 * 250, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, output)
        ).to(device)


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x = x.view(-1, 500, 1)
        return torch.squeeze(x, dim=0)
