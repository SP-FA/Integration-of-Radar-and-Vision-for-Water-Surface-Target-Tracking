from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
from rich.progress import track

from abc import ABC, abstractmethod
from model.pointNet import PointNet
from tools.registration import image_registration, point_cloud_registration
from tools.stupid_tools import remove_outlier, map_label_to_color
from util.load_data import DatasetLoader, NNDatasetLoader
from model.networks import MLP, CNN
from model.DBSCAN import ConditionalDBSCAN


def video2img(path, new_path):
    cap = cv2.VideoCapture(path)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    namelst = os.listdir("./data/3")

    for i in track(range(nframes), description="[blue]Video -> Img", style="white", complete_style="blue"):
        _, frame = cap.read()
        if cv2.waitKey(1) == ord('q'):
            break
        cv2.imwrite(new_path + namelst[i], frame)

    cap.release()


class DrawPointsBase:
    _COLOR_LIST = [
        (255, 0, 0),  # 红色
        (0, 255, 0),  # 绿色
        (0, 0, 255),  # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 品红色
        (0, 255, 255),  # 青色
        (128, 0, 0),  # 深红色
        (0, 128, 0),  # 深绿色
        (0, 0, 128),  # 深蓝色
        (128, 128, 128),  # 灰色
        (0, 0, 0)  # 黑色 / 噪点
    ]
    EPS = 20  # 邻域半径
    MIN_SAMPLES = 4  # 最小邻域点数

    def __init__(self, videoPath, device):
        self._frame = None
        self.vPath = videoPath
        self.dbscan = ConditionalDBSCAN(eps=DrawPointsBase.EPS, min_samples=DrawPointsBase.MIN_SAMPLES)

    def _videoCaptureWriter(self, newPath):
        cap = cv2.VideoCapture(self.vPath)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (w, h)

        out = cv2.VideoWriter(newPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        return cap, out, nframes

    @abstractmethod
    def _choose_method(self, method, i): pass

    def new_video(self, newPath, method):
        cap, out, nframes = self._videoCaptureWriter(newPath)
        totalPoints = []
        totalLabels = []
        totalxyxy = []
        # track(range(nframes), description="[#66CCFF]STEP [purple]1")
        for i in tqdm(range(nframes)):
            _, self._frame = cap.read()
            points = self._choose_method(method, i)
            pointIndex = np.where(points[:, 0] != 0)
            points = points[pointIndex]
            # pcd = remove_outlier(points)
            # points = np.asarray(pcd.points)  # [:, :2]
            labels = self.dbscan.conditional_fit(points, self._frame)
            labels = np.array(labels).astype(int)
            totalxyxy.append(self.dbscan.xyxy)
            labels = map_label_to_color(labels)
            totalPoints.append(points[:, :2].astype(int))
            totalLabels.append(labels)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        pbar = track(range(len(totalPoints)), description="[#66CCFF]STEP [purple]2", style="white", complete_style="#66CCFF")
        for i in pbar:
            _, self._frame = cap.read()
            points = np.array(totalPoints[i]).astype(int)
            labels = np.array(totalLabels[i]).astype(int)
            xyxy = totalxyxy[i]

            if xyxy is not None:
                xyxy = np.array(xyxy).astype(int)
                for j in range(len(xyxy)):
                    p = xyxy[j]
                    cv2.rectangle(self._frame, (p[0], p[1]), (p[2], p[3]), (0, 0, 255), 2)

            if points.shape[0] != 0:
                u = points[:, 0]
                v = points[:, 1]
                for j in range(u.shape[0]):
                    cv2.circle(self._frame, (u[j], v[j]), 1, self._COLOR_LIST[labels[j]], 4)
            out.write(self._frame)
        cap.release()
        out.release()



class NaiveDrawPoints(DrawPointsBase):
    def __init__(self, datasetPath, isCalib=False, fixHeight=False, device=torch.cpu):
        videoPath = os.path.join(datasetPath, "Video.avi")
        super().__init__(videoPath, device)
        self._lastFrame = None
        self.isCalib = isCalib
        self.fixHeight = fixHeight
        self.dl = DatasetLoader(datasetPath)

    def _choose_method(self, method, i):
        """
        :param method:
            - 'none'
            - 'img': 用 Image Registration
            - 'point': 用 Point Cloud Registration
            - 'both': img & point 两个 Registration
        :param i:
        :return: [n, 2]
        """
        if self._lastFrame is None or method == 'none':
            points = self.dl.load2DPoints(i, self.isCalib, self.fixHeight)
        else:
            if   method == 'img':   points = self._img_reg(i)
            elif method == 'point': points = self._point_reg(i)
            elif method == 'none':  points = self.dl.load2DPoints(i, self.isCalib, self.fixHeight)
            elif method == 'both':
                points = self._point_reg(i)
                points = self._img_reg(i, points)
            else: raise ValueError("parameter 'method' error")
        self._lastFrame = self._frame
        return points


    def _img_reg(self, i, points=None):
        if points is None:
            points = self.dl.load2DPoints(i, self.isCalib, self.fixHeight)
        points, _ = image_registration(points, self._frame, self._lastFrame)
        return points


    def _point_reg(self, i):
        source = self.dl.load2DPoints(i, self.isCalib, self.fixHeight)
        target = self.dl.load2DPoints(i - 3, self.isCalib, self.fixHeight)
        return point_cloud_registration(source, target)


class NNDrawPoints(DrawPointsBase):
    def __init__(self, datasetPath, modelPath='weights/mlp_k1.pt', device=torch.device("cpu")):
        videoPath = os.path.join(datasetPath, "Video.avi")
        super().__init__(videoPath, device)
        self.cnn = None
        self.mlp = None
        self.net = None
        self.dataX = None
        self.trueU = None
        self.device = device
        self.modelPath = modelPath
        self.dl = NNDatasetLoader(datasetPath)


    def _choose_method(self, method, i):
        if   method == 'mlp':       points = self._mlp(i)
        elif method == 'cnn':       points = self._cnn(i)
        elif method == 'pointnet':  points = self._pointNet(i)
        else:                       raise ValueError("parameter 'method' error")
        return points


    def _load_data(self, method):
        if self.dataX is None:
            if method == 'mlp':      self.dataX, self.trueU = self.dl.loadValid(False, 1, self.device)
            if method == 'cnn':      self.dataX, self.trueU = self.dl.loadValid(True , 1, self.device)
            if method == 'pointnet': self.dataX, self.trueU = self.dl.loadValid(True , 1, self.device)
        return self.dataX, self.trueU


    def _predict(self, i, model):
        x = self.dataX[i]
        u = self.trueU[i]
        z = x[:, 2].cpu().numpy()
        power = x[:, 3].cpu().numpy()
        doppler = x[:, 5].cpu().numpy()

        y, _, _ = model.predict(x)
        v = y[:, 0].cpu().numpy()
        return np.array([u, v, z, doppler, power]).T

    def _predict_pointnet(self, i, model):
        x = self.dataX[i]
        u = self.trueU[i]
        z = x[:, 2].cpu().numpy()
        power = x[:, 3].cpu().numpy()
        doppler = x[:, 5].cpu().numpy()

        z = z.flatten()
        power = power.flatten()
        doppler = doppler.flatten()

        y, _, _ = model.predict(x)
        y = y.view(-1, 1)
        v = y[:, 0].cpu().numpy()
        return np.array([u, v, z, doppler, power]).T

    def _mlp(self, i):
        """
        使用 mlp 对点云进行校准
        :param i:
        :return: [m, 3]
        """
        if self.mlp is None:
            self.mlp = MLP(modelPath="./weights/mlp_k1.pt", device=self.device)
        return self._predict(i, self.mlp)


    def _cnn(self, i):
        """
        使用 cnn 对点云进行校准
        :param i:
        :return: [500, 3]
        """
        if self.cnn is None:
            self.cnn = CNN(self.dl.input_dim, 500, self.device)
            self.cnn.load_state_dict(torch.load("./weights/cnn_k1.pt"))
        return self._predict(i, self.cnn)


    def _pointNet(self, i):
        if self.net is None:
            self.net = PointNet(modelPath=self.modelPath, device=self.device)
        return self._predict_pointnet(i, self.net)


    def new_video(self, newPath, method):
        self._load_data(method)
        super().new_video(newPath, method)


if __name__ == "__main__":
    new_video_path = './data/pointNet_k1_epoch400_global.avi'

    dp = NNDrawPoints('./data', modelPath="./weights/pointNet_k1_epoch400_global.pt", device=torch.device("cuda"))
    dp.new_video(new_video_path, 'pointnet')
    # dp = NaiveDrawPoints(r'H:\dataset\Tracking\YOLO_timestamp\1', True)
    # dp.new_video(new_video_path, 'none')

    # nimgs = './data/point_img/'
    # video2img(new_video_path, nimgs)

    # path = r'H:\dataset\Tracking\YOLO_timestamp'
    # folders = os.listdir(path)
    # for folder in folders:
    #     if "." in folder: continue
    #     if 10 <= int(folder) <= 16: continue
    #     files = os.listdir(os.path.join(path, folder))
    #     if "Point_Video.avi" in files: continue
    #     print(folder)
    #     datasetPath = os.path.join(path, folder)
    #     dp = NaiveDrawPoints(datasetPath, True)
    #     dp.new_video(os.path.join(path, folder, "Point_Video.avi"), 'none')

