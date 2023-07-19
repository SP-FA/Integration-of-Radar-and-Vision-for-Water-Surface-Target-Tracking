import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from torch.utils.data import TensorDataset, random_split, DataLoader
from tools.calibration import project_pcl_to_image
from model.kdTree import KDTree


IMU = "imu"
CALIB = "calib"
LABEL = "label"
RADAR = "radar_3"


def getIndex(arr, point):
    indexes = np.where((arr[:, 0] == point[0]) & (arr[:, 1] == point[1]) & (arr[:, 2] == point[2]))
    return indexes[0][0]


class DatasetLoader:
    _csvList = None
    _maxPoints = 500

    def __init__(self, datasetPath):
        self._path = datasetPath
        self._radarPath = os.path.join(self._path, RADAR)
        self._imuPath = os.path.join(self._path, IMU)
        self._calib = os.listdir(os.path.join(self._path, CALIB))[0]

        with open(os.path.join(self._path, CALIB, self._calib), "r") as f:
            lines = f.readlines()
            self._extrinsic = np.array(lines[0].strip().split(' ')[1:], dtype=np.float32).reshape(4, 4)
            self._intrinsic = np.array(lines[1].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)


    @property
    def csv(self):
        if DatasetLoader._csvList is None:
            DatasetLoader._csvList = os.listdir(os.path.join(self._path, IMU))
        return DatasetLoader._csvList


    def loadMatrix(self):
        return self._extrinsic, self._intrinsic


    def loadCSV(self, name, i):
        if isinstance(i, int):
            frame = self.csv[i]
            return pd.read_csv(os.path.join(self._path, name, frame), index_col=0)
        else:
            return pd.read_csv(os.path.join(self._path, name, i), index_col=0)


    def load3DPoints(self, i):
        """
        :param i:
        :return: [n, 3]
        """
        frame = self.csv[i]
        df = pd.read_csv(os.path.join(self._path, RADAR, frame), index_col=0)
        return np.array([df['x'], df['y'], df['z']]).T


    def load2DPoints(self, i, isCalib, fixHeight):
        """
        :param i:
        :param isCalib: Whether the points need to be calibrated.
        :param fixHeight: Whether the fix-height method is used to calibration.
                          It will only take effect when isCalib is True.
        :return: [n, 2]
        """
        if isCalib is True:
            return project_pcl_to_image(self, i, fixHeight)
        else:
            frame = self.csv[i]
            df = pd.read_csv(os.path.join(self._path, RADAR, frame), index_col=0)
            return np.array([df['u'], df['v']]).T


    def loadCompH(self, i):
        frame = self.csv[i]
        df = pd.read_csv(os.path.join(self._path, RADAR, frame), index_col=0)
        return np.array([df['comp_height']])


    def loadPitch(self, i):
        frame = self.csv[i]
        df = pd.read_csv(os.path.join(self._path, IMU, frame), index_col=0)
        return df['pitch'].tolist()[0]


class NNDatasetLoader(DatasetLoader):
    def __init__(self, datasetPath):
        super().__init__(datasetPath)
        self._labelPath = os.path.join(self._path, LABEL)
        self._k = None


    @property
    def input_dim(self):
        """
        Call method loadTrainTest or loadValid at first.
        :return:
        """
        return self._k * 9 + 9


    @property
    def output_dim(self):
        return 1


    def _get_features(self, radarcsv, imucsv):
        """

        :param radarcsv:
        :param imucsv:
        :return: [m, k * 11 + 9]
        """
        lenth = radarcsv.shape[0]
        x = radarcsv["x"].tolist()
        y = radarcsv["y"].tolist()
        z = radarcsv["z"].tolist()
        rang = radarcsv["range"].tolist()
        doppler = radarcsv["doppler"].tolist()
        azimuth = radarcsv["azimuth"].tolist()
        elevation = radarcsv["elevation"].tolist()
        comp_height = radarcsv["comp_height"].tolist()
        comp_velocity = radarcsv["comp_velocity"].tolist()

        pitch = imucsv["pitch"].tolist()[0]
        roll = imucsv["roll"].tolist()[0]
        yaw = imucsv["yaw"].tolist()[0]
        avx = imucsv["angular_velocity_x"].tolist()[0]
        avy = imucsv["angular_velocity_y"].tolist()[0]
        avz = imucsv["angular_velocity_z"].tolist()[0]
        lvx = imucsv["linear_acceleration_x"].tolist()[0]
        lvy = imucsv["linear_acceleration_y"].tolist()[0]
        lvz = imucsv["linear_acceleration_z"].tolist()[0]

        if self._k == 1:
            datax = np.array([
                x, y, z, rang, doppler, azimuth, elevation, comp_height, comp_velocity,
                [pitch for _ in range(lenth)], [roll for _ in range(lenth)], [yaw for _ in range(lenth)],
                [avx for _ in range(lenth)], [avy for _ in range(lenth)], [avz for _ in range(lenth)],
                [lvx for _ in range(lenth)], [lvy for _ in range(lenth)], [lvz for _ in range(lenth)],
            ]).T
        else:
            datax = []
            imuVector = np.array([pitch, roll, yaw, avx, avy, avz, lvx, lvy, lvz])
            points = np.array([x, y, z]).T  # [m, 3]
            fullRadars = np.array([x, y, z, rang, doppler, azimuth, elevation, comp_height, comp_velocity]).T
            kdt = KDTree(points, self._k)
            for i in range(points.shape[0]):
                pointVector = np.empty(0)  # [1]
                nps, _ = kdt.search_nearest(i, isInner=True)
                if len(nps) < self._k:
                    pointVector = np.zeros((self._k - len(nps)) * 11)
                for j in nps:
                    index = getIndex(points, j)
                    pointVector = np.concatenate((pointVector, fullRadars[index]))  # [k * 11]
                pointVector = np.concatenate((pointVector, imuVector))
                datax.append(pointVector)
            datax = np.array(datax)
        return datax


    def _padding_frames(self, data):
        """
        把这一帧的点云数量 padding 到 self._maxPoints
        :param data: 要 padding 的点云数据
        :return: [maxPoints, input_dim], ndarray
        """
        if data.shape[0] < self._maxPoints:
            diff = self._maxPoints - data.shape[0]
            zeros = np.zeros([diff, data.shape[1]])
            data = np.concatenate((data, zeros), axis=0)
        if data.shape[0] > self._maxPoints:
            data = data[:self._maxPoints]
        return data


    def _dataX(self, device=torch.device("cpu")):
        """
        :return: [n, input_dim], torch.tensor
        """
        data_x = np.empty([1, self.input_dim])
        csv = os.listdir(self._labelPath)
        pbar = tqdm(csv)
        pbar.set_description("[Loading data]")
        for file in pbar:
            radarcsv = self.loadCSV(RADAR, file)
            imucsv = self.loadCSV(IMU, file)
            datax = self._get_features(radarcsv, imucsv)
            data_x = np.concatenate((data_x, datax), axis=0)
        return torch.tensor(data_x).to(torch.float32).to(device)


    def _dataY(self, device=torch.device("cpu")):
        """
        :return: [n, output_dim], torch.tensor
        """
        data_y = np.empty([1, 1])
        csv = os.listdir(self._labelPath)
        for i in csv:
            labelcsv = self.loadCSV(LABEL, i)
            v = labelcsv["v"].tolist()
            datay = np.array([v]).T
            data_y = np.concatenate((data_y, datay), axis=0)
            data_y = np.nan_to_num(data_y, nan=0)
        return torch.tensor(data_y).to(torch.float32).to(device)


    def _frameX(self, isValid=False, padding=False, device=torch.device("cpu")):
        """
        :param isValid: 如果是训练模式，就选取标注过的数据，如果是验证模式，则选取所有数据
        :param padding: 是否进行 padding
        :param device:
        :return: 如果 padding 了，那么返回的形状是 [nframe, 500, input_dim]，否则返回一个 list，其中的每一个元素是一个 tensor [m, input_dim]
        """
        data_x = []
        csv = self.csv if isValid else os.listdir(self._labelPath)
        pbar = tqdm(csv)
        pbar.set_description("[Loading data]")
        for i in pbar:
            radarcsv = self.loadCSV(RADAR, i)
            imucsv = self.loadCSV(IMU, i)
            datax = self._get_features(radarcsv, imucsv)
            datax = self._padding_frames(datax) if padding else torch.tensor(datax).to(torch.float32).to(device)
            data_x.append(datax)
        if padding:
            data_x = np.array(data_x)
            return torch.tensor(data_x).to(torch.float32).to(device)
        else:
            return data_x


    def _frameY(self, isValid=False, padding=False, device=torch.device("cpu")):
        data_y = []
        csv = self.csv if isValid else os.listdir(self._labelPath)
        for i in csv:
            labelcsv = self.loadCSV(LABEL, i)
            v = labelcsv["v"].tolist()
            datay = np.array([v]).T  # [m, 1]
            datay = self._padding_frames(datay) if padding else torch.tensor(datay).to(device)
            data_y.append(datay)
        if padding:
            data_y = np.array(data_y)
            return torch.tensor(data_y).to(torch.float32).to(device)  # [n, 500, 1]
        else:
            return data_y


    def loadTrainTest(self, batch=218, sep=0.9, k=5, padding=False, device=torch.device("cpu")):
        self._k = k
        if padding is True:
            data_x = self._frameX(False, padding, device)
            data_x = data_x.permute(0, 2, 1)  # 卷积层第二维是 channel
            data_y = self._frameY(False, padding, device)
        else:
            data_x = self._dataX(device)
            data_y = self._dataY(device)

        data_dataset = TensorDataset(data_x, data_y)
        train_size = int(len(data_dataset) * sep)
        test_size = len(data_dataset) - train_size
        train_data, test_data = random_split(data_dataset, [train_size, test_size])
        trainSet = DataLoader(train_data, batch_size=batch)
        testSet = DataLoader(test_data, batch_size=batch)
        return trainSet, testSet


    def loadValid(self, padding=False, k=5, device=torch.device("cpu")):
        self._k = k
        valid = self._frameX(True, padding, device)
        if padding:
            valid = valid.permute(0, 2, 1)
            valid = torch.unsqueeze(valid, dim=1)
        us = []
        csv = self.csv
        for i in csv:
            radarcsv = self.loadCSV(RADAR, i)
            u = radarcsv["u"].tolist()  # [1, m]
            if padding:
                if len(u) < self._maxPoints:
                    u.extend([0 for i in range(self._maxPoints - len(u))])
                if len(u) > self._maxPoints:
                    u = u[:self._maxPoints]
            us.append(u)
        return valid, us  # [nframe, m, 20], [nframe, m]
