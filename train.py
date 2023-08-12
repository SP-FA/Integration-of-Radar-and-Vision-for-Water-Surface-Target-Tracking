import torch
from model.cnn import MLP
from model.pointNet import PointNet

if __name__ == "__main__":
    epoch = 400
    device = torch.device("cuda")

    # k=1 [19, 32, 32, 8, 1]
    # k=2 [  , 32, 32, 8, 1]
    # k=5 [  , 64, 32, 8, 1]
    # model = MLP("../cfg/mlp.json", device)
    # model.train("./data", epoch, "./weights/mlp_k1.pt")

    # model = CNN(20, 500, device)
    # train(model, trainSet, testSet, epoch, "cnn.pt", device=device)

    model = PointNet(config="./cfg/pointNet.json", device=device)
    # model.train("./data", epoch, "./weights/pointNet_k1_epoch400.pt", "pointNet_k1_epoch400.jpg")
    # model.train("./data", epoch, "./weights/pointNet_k3_epoch400.pt", "pointNet_k3_epoch400.jpg")
    # model.train("./data", epoch, "./weights/pointNet_k5_epoch400.pt", "pointNet_k5_epoch400.jpg")

    model.train("./point_calib_dataset", epoch, "./weights/pointNet_k1_epoch400_global.pt", "pointNet_k1_epoch400_global.jpg")

