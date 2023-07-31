from rich.progress import track
from abc import ABC, abstractmethod
from tools.visualization import draw_plots
from util.load_cfg import ConfigureLoader
import torch


class NetBase(ABC):
    def __init__(self, kwargs):
        if "config" in kwargs: self.cl = ConfigureLoader(kwargs["config"])
        if "device" in kwargs: self.D = kwargs["device"]
        self.model = None

    def predict(self, x):
        self.model.eval()
        with torch.no_grad(): return self.model(x)

    def train(self, datasetPath, epoch, savePath, lr=1e-3):
        train, test = self._load_data(datasetPath)
        optimizer = self._optim(lr)
        train_loss_list = []
        test_loss_list = []

        for i_epoch in range(epoch):
            totalTrainLoss = 0
            totalTestLoss = 0

            self.model.train()
            pbar = track(train, description="Train %d/%d" % (i_epoch, epoch), style="white", complete_style="blue")
            for x, y in pbar:
                optimizer.zero_grad()
                pred = self.model(x)

                loss = self._loss(pred, y)
                loss.backward()
                optimizer.step()

                totalTrainLoss += loss.item()
            avgTrainLoss = totalTrainLoss / len(train)

            self.model.eval()
            pbar = track(test, description="Test  %d/%d" % (i_epoch, epoch), style="white", complete_style="blue")
            for x, y in pbar:
                optimizer.zero_grad()
                pred = self.model(x)

                loss = self._loss(pred, y)
                totalTestLoss += loss.item()
            avgTestLoss = totalTestLoss / len(test)

            train_loss_list.append(avgTrainLoss)
            test_loss_list.append(avgTestLoss)
        draw_plots("Loss.jpg", train_loss_list, test_loss_list)
        torch.save(self.model, savePath)

    @property
    def input_dim(self): return self.cl.k * 10 + 9
    @property
    def output_dim(self): return 1

    @abstractmethod
    def _load_data(self, datasetPath): pass

    @abstractmethod
    def _optim(self, lr): pass

    @abstractmethod
    def _loss(self, pred, y): pass
