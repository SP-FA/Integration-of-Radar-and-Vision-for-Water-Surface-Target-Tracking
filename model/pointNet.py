import torch
from torch import nn, optim
import torch.nn.functional as F

from model.netBase import NetBase
from tools.algorithm import feature_trans_regularizer
from util.load_data import NNDatasetLoader


class PointNet(NetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "modelPath" in kwargs: self.model = torch.load(kwargs["modelPath"])
        else: self.model = PointNetSeg(self.input_dim, self.output_dim, self.cl.globalFeature, self.cl.useFeatureTrans, self.D)

    def _load_data(self, datasetPath):
        dl = NNDatasetLoader(datasetPath)
        return dl.loadTrainTest(self.cl.batch, 0.9, self.cl.k, padding=True, maxPoints=self.cl.maxPoints, device=self.D)

    def _optim(self, lr):
        return optim.Adam(params=self.model.parameters(), lr=lr)

    def _loss(self, predY, y):
        lossFunc = nn.MSELoss().to(self.D)
        pred, _, featureTrans = predY

        loss = lossFunc(pred, y)
        if self.cl.useFeatureTrans:
            loss += feature_trans_regularizer(featureTrans) * 0.001
        return loss


class STN3d(nn.Module):
    def __init__(self, input_dim, device=torch.device("cpu")):
        super(STN3d, self).__init__()
        self.input = input_dim

        self.conv1 = nn.Conv1d(input_dim, 64, 1).to(device)
        self.conv2 = nn.Conv1d(64, 128, 1).to(device)
        self.conv3 = nn.Conv1d(128, 1024, 1).to(device)
        self.fc1 = nn.Linear(1024, 512).to(device)
        self.fc2 = nn.Linear(512, 256).to(device)
        self.fc3 = nn.Linear(256, input_dim * input_dim).to(device)
        self.relu = nn.ReLU().to(device)

        self.bn1 = nn.BatchNorm1d(64).to(device)
        self.bn2 = nn.BatchNorm1d(128).to(device)
        self.bn3 = nn.BatchNorm1d(1024).to(device)
        self.bn4 = nn.BatchNorm1d(512).to(device)
        self.bn5 = nn.BatchNorm1d(256).to(device)
        self.device = device

    def forward(self, x):
        batchSize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.input, dtype=torch.float32).view(1, self.input * self.input).repeat(batchSize, 1)
        if x.is_cuda: iden = iden.cuda()

        x = x + iden
        x = x.view(-1, self.input, self.input)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64, device=torch.device("cpu")):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1).to(device)
        self.conv2 = torch.nn.Conv1d(64, 128, 1).to(device)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1).to(device)
        self.fc1 = nn.Linear(1024, 512).to(device)
        self.fc2 = nn.Linear(512, 256).to(device)
        self.fc3 = nn.Linear(256, k * k).to(device)
        self.relu = nn.ReLU().to(device)

        self.bn1 = nn.BatchNorm1d(64).to(device)
        self.bn2 = nn.BatchNorm1d(128).to(device)
        self.bn3 = nn.BatchNorm1d(1024).to(device)
        self.bn4 = nn.BatchNorm1d(512).to(device)
        self.bn5 = nn.BatchNorm1d(256).to(device)
        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, dtype=torch.float32).view(1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda: iden = iden.cuda()

        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    def __init__(self, input_dim, globalFeature=True, useFeatureTrans=False, device=torch.device("cpu")):
        super(PointNetFeat, self).__init__()
        self.stn = STN3d(input_dim, device)
        self.conv1 = torch.nn.Conv1d(input_dim, 64, 1).to(device)
        self.conv2 = torch.nn.Conv1d(64, 128, 1).to(device)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1).to(device)
        self.bn1 = nn.BatchNorm1d(64).to(device)
        self.bn2 = nn.BatchNorm1d(128).to(device)
        self.bn3 = nn.BatchNorm1d(1024).to(device)
        self.globalFeature = globalFeature
        self.useFeatureTrans = useFeatureTrans
        if self.useFeatureTrans: self.fstn = STNkd(64, device)

    def forward(self, x):
        n_pts = x.size()[2]
        inputTrans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, inputTrans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.useFeatureTrans:
            featureTrans = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, featureTrans)
            x = x.transpose(2, 1)
        else:
            featureTrans = None

        pointFeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.globalFeature:
            return x, inputTrans, featureTrans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointFeat], 1), inputTrans, featureTrans


class PointNetSeg(nn.Module):
    def __init__(self, input_dim, output_dim, globalFeature=False, useFeatureTrans=False, device=torch.device("cpu")):
        super(PointNetSeg, self).__init__()
        self.input = input_dim
        self.output = output_dim

        self.feature_transform = useFeatureTrans
        self.feat = PointNetFeat(input_dim, globalFeature, useFeatureTrans, device)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1).to(device)
        self.conv2 = torch.nn.Conv1d(512, 256, 1).to(device)
        self.conv3 = torch.nn.Conv1d(256, 128, 1).to(device)
        self.conv4 = torch.nn.Conv1d(128, self.output, 1).to(device)
        self.bn1 = nn.BatchNorm1d(512).to(device)
        self.bn2 = nn.BatchNorm1d(256).to(device)
        self.bn3 = nn.BatchNorm1d(128).to(device)

    def forward(self, x):
        batchSize = x.size()[0]
        n_pts = x.size()[2]
        x, inputTrans, featureTrans = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = x.view(-1, self.output)
        x = x.view(batchSize, n_pts, self.output)
        return x, inputTrans, featureTrans
