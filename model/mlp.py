import torch
from torch import optim, nn

from model.netBase import NetBase
from util.load_data import NNDatasetLoader


class MLP(NetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "modelPath" in kwargs: self.model = torch.load(kwargs["modelPath"])
        else:
            layers = self.cl.layers
            layers.insert(0, self.input_dim)
            layers.append(self.output_dim)
            self.model = MLP_model(layers, self.D)

    def _load_data(self, datasetPath):
        dl = NNDatasetLoader(datasetPath)
        return dl.loadTrainTest(self.cl.batch, 0.9, self.cl.k, padding=False, device=self.D)

    def _optim(self, lr):
        return optim.Adam(params=self.model.parameters(), lr=lr)

    def _loss(self, predY, y):
        lossFunc = nn.MSELoss().to(self.D)
        return lossFunc(predY, y)


class MLP_model(nn.Module):
    def __init__(self, layers, device):
        super(MLP_model, self).__init__()
        self.layers = nn.Sequential().to(device)
        for i in range(len(layers) - 2):
            linear = nn.Linear(layers[i], layers[i + 1]).to(device)
            batchnorm = nn.BatchNorm1d(layers[i + 1]).to(device)
            relu = nn.ReLU(inplace=True).to(device)
            nn.init.kaiming_uniform_(linear.weight, a=0, mode='fan_in', nonlinearity='relu')

            self.layers.add_module("Linear%d" % i, linear)
            self.layers.add_module("BatchNorm%d" % i, batchnorm)
            self.layers.add_module("ReLU%d" % i, relu)

        linear = nn.Linear(layers[-2], layers[-1]).to(device)
        nn.init.kaiming_uniform_(linear.weight, a=0, mode='fan_in', nonlinearity='relu')
        self.layers.add_module("Linear%d" % (len(layers) - 2), linear)
        self.device = device

    def forward(self, x):
        x = self.layers(x)
        nan_mask = torch.isnan(x)
        x[nan_mask] = 0
        return x
