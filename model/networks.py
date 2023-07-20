import torch
from torch import optim, nn
from tqdm import tqdm

from tools.visualization import draw_plots
from util.load_cfg import ConfigureLoader
from util.load_data import NNDatasetLoader


class MLP:
    def __init__(self, config, device=torch.device("cpu")):
        self.cl = ConfigureLoader(config)
        self.layers = self.cl.layers
        self.layers.insert(0, self.input_dim)
        self.layers.append(self.output_dim)
        self.model = MLP_model(self.layers, device)
        self.device = device
        self._isLoaded = False

    @property
    def input_dim(self): return self.cl.k * 10 + 9

    @property
    def output_dim(self): return 1

    def train(self, datasetPath, epoch, savePath, lr=1e-2):
        dl = NNDatasetLoader(datasetPath)
        train, test = dl.loadTrainTest(self.cl.batch, 0.95, k=self.cl.k, padding=False, device=self.device)
        optimizer = optim.Adam(params=self.model.parameters(), lr=lr)
        loss_func = nn.MSELoss().to(self.device)
        trainSize = len(train)
        testSize = len(test)
        train_loss_list = []
        val_loss_list = []

        for i_epoch in range(epoch):
            totalTrainLoss = 0
            totalTestLoss = 0

            self.model.train()
            trainBar = tqdm(train)
            for x, y in trainBar:
                optimizer.zero_grad()
                pred = self.model(x)

                loss = loss_func(pred, y)
                loss.backward()
                optimizer.step()

                totalTrainLoss += loss.item()
                trainBar.set_description("[Train][Epoch: %d][loss: %d]" % (i_epoch, loss))
            avgTrainLoss = totalTrainLoss / trainSize

            self.model.eval()
            testBar = tqdm(test)
            for x, y in testBar:
                optimizer.zero_grad()
                pred = self.model(x)
                loss = loss_func(pred, y)
                totalTestLoss += loss.item()
                testBar.set_description("[Test ][Epoch: %d][loss: %d]" % (i_epoch, loss))
            avgTestLoss = totalTestLoss / testSize

            train_loss_list.append(avgTrainLoss)
            val_loss_list.append(avgTestLoss)
        draw_plots("Loss&acc.jpg", train_loss_list[30:], val_loss_list[30:])
        torch.save(self.model.state_dict(), savePath)


    def predict(self, x, modelPath):
        if not self._isLoaded:
            self.model.load_state_dict(torch.load(modelPath))

        self.model.eval()
        with torch.no_grad():
            return self.model(x)


class MLP_model(nn.Module):
    def __init__(self, layers, device):
        super(MLP_model, self).__init__()
        self.layers = nn.Sequential().to(device)
        for i in range(len(layers) - 2):
            linear = nn.Linear(layers[i], layers[i + 1]).to(device)
            batchnorm = nn.BatchNorm1d(layers[i + 1]).to(device)
            relu = nn.ReLU(inplace=True).to(device)
            nn.init.kaiming_uniform_(linear.weight, a=0, mode='fan_in', nonlinearity='relu')

            self.layers.add_module("Linear%d" % (i), linear)
            self.layers.add_module("BatchNorm%d" % (i), batchnorm)
            self.layers.add_module("ReLU%d" % (i), relu)

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
