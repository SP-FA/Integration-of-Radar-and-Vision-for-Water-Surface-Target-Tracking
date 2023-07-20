import torch
from model.networks import MLP


if __name__ == "__main__":
    epoch = 200
    device = torch.device("cuda")

    # k=1 [19, 32, 32, 8, 1]
    # k=2 [  , 32, 32, 8, 1]
    # k=5 [  , 64, 32, 8, 1]
    model = MLP("../cfg/mlp.json", device)
    model.train("./data", epoch, "./weights/mlp_k1.pt")

    # model = CNN(20, 500, device)
    # train(model, trainSet, testSet, epoch, "cnn.pt", device=device)
