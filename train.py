from torch import optim, nn
from tqdm import tqdm
import torch

from model.networks import MLP
from util import DatasetLoader
from tools.visualization import draw_plots


def train(mlp, train, test, epoch, save, device=torch.device("cpu")):
    optimizer = optim.Adam(params=mlp.parameters(), lr=1e-2)
    loss_func = nn.MSELoss().to(device)
    train_loss_list = []
    val_loss_list = []

    for i_epoch in range(epoch):
        total_loss = 0
        total_val_loss = 0

        epoch_size_trn = len(train)
        epoch_size_val = len(test)

        train_pbar = tqdm(train)
        mlp.train()
        for x, y in train_pbar:
            optimizer.zero_grad()
            pred = mlp(x)
            # print(pred, y)
            loss = loss_func(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_pbar.set_description("[Train][Epoch: %d][loss: %d]" % (i_epoch, loss))

        train_loss = total_loss / epoch_size_trn

        test_pbar = tqdm(test)
        mlp.eval()
        for x, y in test_pbar:
            optimizer.zero_grad()
            pred = mlp(x)
            loss = loss_func(pred, y)
            total_val_loss += loss.item()
            test_pbar.set_description("[Test ][Epoch: %d][loss: %d]" % (i_epoch, loss))

        val_loss = total_val_loss / epoch_size_val
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

    draw_plots("Loss&acc.jpg", epoch-30, train_loss_list[30:], val_loss_list[30:])
    torch.save(mlp.state_dict(), save)


if __name__ == "__main__":
    epoch = 200
    device = torch.device("cuda")
    batch_size = 512

    dl = DatasetLoader("./data")
    trainSet, testSet = dl.loadTrainTest(batch_size, 0.95, k=1, padding=False, device=device)
    layers = [dl.input_dim, 32, 32, 16, dl.output_dim]  # k=1 [20, 32, 32, 8, 1]
                                                        # k=2 [31, 32, 32, 8, 1]
                                                        # k=5 [  , 64, 32, 8, 1]
    model = MLP(layers, device)
    train(model, trainSet, testSet, epoch, "mlp.pt", device=device)

    # model = CNN(20, 500, device)
    # train(model, trainSet, testSet, epoch, "cnn.pt", device=device)
