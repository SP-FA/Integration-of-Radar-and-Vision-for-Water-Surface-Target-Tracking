from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
from rich.progress import track

from abc import ABC, abstractmethod

from ultralytics import YOLO

from model.cnn import CNN_model, CNN
from model.pointNet import PointNet
from tools.registration import image_registration, point_cloud_registration
from tools.stupid_tools import remove_outlier, map_label_to_color
from util.load_cfg import ConfigureLoader
from util.load_data import DatasetLoader, NNDatasetLoader, RADAR
# from model.cnn import CNN
from model.mlp import MLP
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


class DrawPointsBase(ABC):
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
        self.yolo = YOLO("../weights/best.pt")
        self.dl = None

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

    def new_video(self, newPath, method, add=False):
        cap, out, nframes = self._videoCaptureWriter(newPath)
        results = self.yolo.track(source=self.vPath, verbose=False)
        totalPoints = []
        totalLabels = []
        totalXYXY = []
        totalID = []
        totalCLS = []
        pbar = track(range(nframes), description="[blue]STEP [purple]1", style="white", complete_style="blue")
        for i in pbar:
            _, self._frame = cap.read()
            points = self._choose_method(method, i)
            # pcd = remove_outlier(points)
            # points = np.asarray(pcd.points)  # [:, :2]
            result = results[i].boxes
            id = result.id
            cls = result.cls
            xyxy = result.xyxy
            conf = result.conf

            if id is not None:
                sortedIndex = conf.argsort(descending=True)
                xyxy = xyxy[sortedIndex]
                cls = cls[sortedIndex]
                id = id[sortedIndex]

            labels = self.dbscan.conditional_fit(points, id, xyxy)
            labels = np.array(labels).astype(int)
            totalXYXY.append(xyxy)
            totalID.append(id)
            totalCLS.append(cls)
            totalPoints.append(points[:, :2].astype(int))
            totalLabels.append(labels)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        pbar = track(range(nframes), description="[blue]STEP [purple]2", style="white", complete_style="blue")
        for i in pbar:
            _, self._frame = cap.read()
            points = np.array(totalPoints[i]).astype(int)
            labels = np.array(totalLabels[i]).astype(int)
            xyxy = totalXYXY[i]
            cls = totalCLS[i]
            ids = totalID[i]

            if ids is not None:
                xyxy = xyxy.cpu()
                cls = cls.cpu()
                ids = ids.cpu()
                xyxy = np.array(xyxy).astype(int)
                for j in range(len(xyxy)):
                    p = xyxy[j]
                    cv2.rectangle(self._frame, (p[0], p[1]), (p[2], p[3]), (0, 0, 255), 2)
                    # 图像，文字内容，坐标(右上角坐标) ，字体，大小，颜色，字体厚度
                    cv2.putText(self._frame, 'id:%d' % ids[j], (p[0], p[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 1)

            if points.shape[0] != 0:
                u = points[:, 0]
                v = points[:, 1]
                clss = []
                idss = []
                for j in range(u.shape[0]):
                    if labels[j] != 0:
                        idss.append(int(ids[labels[j] - 1]))
                        clss.append(int(cls[labels[j] - 1]))
                    else:
                        idss.append(-1)
                        clss.append(-1)
                    cv2.circle(self._frame, (u[j], v[j]), 1, self._COLOR_LIST[labels[j]], 4)
                if add: self.dl.changeCSV(RADAR, i, calib_u=u, calib_v=v, id=idss, cls=clss)
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
    def __init__(self, datasetPath, modelPath, config, device=torch.device("cpu")):
        videoPath = os.path.join(datasetPath, "Video.avi")
        super().__init__(videoPath, device)
        self.model = None
        self.dataX = None
        self.trueU = None
        self.D = device
        self.modelPath = modelPath
        self.dl = NNDatasetLoader(datasetPath)
        self.cl = ConfigureLoader(config)


    def _choose_method(self, method, i):
        if method in ['mlp', 'cnn', 'pointNet']:
            points = self._predict(i, method)
            points = points[:self.rawLen[i]]
            return points
        else:
            raise ValueError("parameter 'method' error")


    def _load_data(self, method):
        if self.dataX is None:
            if method == 'mlp':      self.dataX, self.trueU, self.rawLen = self.dl.loadValid(False, self.cl.maxPoints, self.cl.k, self.D)
            if method == 'cnn':      self.dataX, self.trueU, self.rawLen = self.dl.loadValid(True,  self.cl.maxPoints, self.cl.k, self.D)
            if method == 'pointNet': self.dataX, self.trueU, self.rawLen = self.dl.loadValid(True,  self.cl.maxPoints, self.cl.k, self.D)
        return self.dataX, self.trueU


    def _predict(self, i, method):
        if self.model is None:
            if method == "mlp":      self.model = MLP(modelPath=self.modelPath, device=self.D)
            if method == "cnn":      self.model = CNN(modelPath=self.modelPath, device=self.D)
            if method == "pointNet": self.model = PointNet(modelPath=self.modelPath, device=self.D)

        x = self.dataX[i]
        u = self.trueU[i]
        z = x[:, 2].cpu().numpy()
        power = x[:, 3].cpu().numpy()
        doppler = x[:, 5].cpu().numpy()

        if method == "pointNet":
            z = z.flatten()
            power = power.flatten()
            doppler = doppler.flatten()

        y, _, _ = self.model.predict(x)
        y = y.view(-1, 1)
        v = y[:, 0].cpu().numpy()
        return np.array([u, v, z, doppler, power]).T


    def new_video(self, newPath, method=None, add=False):
        method = self.cl.model
        self._load_data(method)
        super().new_video(newPath, method, add)


if __name__ == "__main__":
    new_video_path = './data/pointNet_k1_epoch400_global.avi'

    # dp = NNDrawPoints('./data', modelPath="./weights/pointNet_k1_epoch400_global.pt", config="./cfg/pointNet.json", device=torch.device("cuda"))
    # dp.new_video(new_video_path, add=True)
    # dp = NaiveDrawPoints(r'H:\dataset\Tracking\YOLO_timestamp\1', True)
    # dp.new_video(new_video_path, 'none')

    # nimgs = './data/point_img/'
    # video2img(new_video_path, nimgs)

    # err = [9, 10, 11, 12, 13, 14, 15, 16, 43]
    # path = r'H:\dataset\Tracking\YOLO_timestamp'
    # folders = os.listdir(path)
    # for folder in folders:
    #     if "." in folder: continue
    #     if int(folder) in err: continue
    #     if not int(folder) == 47: continue  # 47,
    #     files = os.listdir(os.path.join(path, folder))
    #     # if "Calib_Point_Video.avi" in files: continue
    #     print(folder)
    #     datasetPath = os.path.join(path, folder)
    #     dp = NNDrawPoints(datasetPath, modelPath="./weights/pointNet_k1_epoch400_global.pt", config="./cfg/pointNet.json", device=torch.device("cuda"))
    #     dp.new_video(os.path.join(path, folder, "Calib_Point_Video.avi"), add=True)

