import torch
from torch import optim, nn

from model.netBase import NetBase
from util.load_data import NNDatasetLoader


class CNN(NetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "modelPath" in kwargs: self.model = torch.load(kwargs["modelPath"])
        else:
            self.model = CNN_model(self.input_dim, self.output_dim, self.D)

    def _load_data(self, datasetPath):
        dl = NNDatasetLoader(datasetPath)
        return dl.loadTrainTest(self.cl.batch, 0.9, self.cl.k, padding=True, maxPoints=self.cl.maxPoints, device=self.D)

    def _optim(self, lr):
        return optim.Adam(params=self.model.parameters(), lr=lr)

    def _loss(self, predY, y):
        lossFunc = nn.MSELoss().to(self.D)
        return lossFunc(predY, y)


class CNN_model(nn.Module):
    def __init__(self, input, output, device):
        super(CNN_model, self).__init__()
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
