import torch
from torch import optim, nn

from model.NetBase import NetBase
from util.load_data import NNDatasetLoader


class MLP(NetBase):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        if "modelPath" in kwargs: self.model = torch.load(kwargs["modelPath"])
        else:
            layers = self.cl.layers
            layers.insert(0, self.input_dim)
            layers.append(self.output_dim)
            self.model = MLP_model(layers, self.D)

    def _load_data(self, datasetPath):
        dl = NNDatasetLoader(datasetPath)
        return dl.loadTrainTest(self.cl.batch, 0.95, k=self.cl.k, padding=False, device=self.D)

    def _optim(self, lr):
        return optim.Adam(params=self.model.parameters(), lr=lr)

    def _loss(self, predY, y):
        lossFunc = nn.MSELoss().to(self.D)
        return lossFunc(predY, y)

    # def train(self, datasetPath, epoch, savePath, lr=1e-2):
    #     dl = NNDatasetLoader(datasetPath)
    #     train, test = dl.loadTrainTest(self.cl.batch, 0.95, k=self.cl.k, padding=False, device=self.D)
    #     optimizer = optim.Adam(params=self.model.parameters(), lr=lr)
    #     lossFunc = nn.MSELoss().to(self.D)
    #     train_loss_list = []
    #     test_loss_list = []
    #
    #     for i_epoch in range(epoch):
    #         totalTrainLoss = 0
    #         totalTestLoss = 0
    #
    #         self.model.train()
    #         pbar = track(train, description="Train---- %d/%d" % (i_epoch, epoch), style="white", complete_style="blue")
    #         for x, y in pbar:
    #             optimizer.zero_grad()
    #             pred = self.model(x)
    #
    #             loss = lossFunc(pred, y)
    #             loss.backward()
    #             optimizer.step()
    #
    #             totalTrainLoss += loss.item()
    #         avgTrainLoss = totalTrainLoss / len(train)
    #
    #         self.model.eval()
    #         pbar = track(train, description="Test----- %d/%d" % (i_epoch, epoch), style="white", complete_style="blue")
    #         for x, y in pbar:
    #             optimizer.zero_grad()
    #             pred = self.model(x)
    #             loss = lossFunc(pred, y)
    #             totalTestLoss += loss.item()
    #         avgTestLoss = totalTestLoss / len(test)
    #
    #         train_loss_list.append(avgTrainLoss)
    #         test_loss_list.append(avgTestLoss)
    #     draw_plots("Loss&acc.jpg", train_loss_list[30:], test_loss_list[30:])
    #     torch.save(self.model.state_dict(), savePath)


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
