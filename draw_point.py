from tqdm import tqdm
import numpy as np
import torch
import cv2
import os

from registration import image_registration, point_cloud_registration
from tools.stupid_tools import remove_outlier, map_label_to_color
from util import DatasetLoader, NNDatasetLoader
from model.networks import MLP, CNN
from model.DBSCAN import DBSCAN
from model.yoloIter import YOLOIterator


def video2img(path, new_path):
    cap = cv2.VideoCapture(path)  # 开启摄像头或者读取视频文件
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    namelst = os.listdir("./data/3")

    for i in tqdm(range(nframes)):
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
    EPS = 35  # 邻域半径
    MIN_SAMPLES = 4  # 最小邻域点数

    def __init__(self, datasetPath, videoPath):
        self._frame = None
        self.vPath = videoPath
        self.dbscan = DBSCAN(eps=DrawPointsBase.EPS, min_samples=DrawPointsBase.MIN_SAMPLES)
        self.yolo = YOLOIterator("./yolov8/yolov8_best.pt")


    def _videoCaptureWriter(self, newPath):
        cap = cv2.VideoCapture(self.vPath)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (w, h)

        out = cv2.VideoWriter(newPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        return cap, out, nframes


    def _postprocessing(self, points, pointlst, labellst):
        pcd = remove_outlier(points)
        points = np.asarray(pcd.points)  # [:, :2]
        labels = self.dbscan.fit(points)
        labels = map_label_to_color(labels)
        pointlst.append(points[:, :2].astype(int))
        labellst.append(labels)
        return pointlst, labellst


    def _choose_method(self, method, i):
        """
        You should implement this method at every child class
        """
        raise RuntimeError("You should implement this method at every child class")


    def new_video(self, newPath, method):
        cap, out, nframes = self._videoCaptureWriter(newPath)
        totalPoints = []
        totalLabels = []
        for i in range(nframes):
            _, self._frame = cap.read()
            xyxy, conf = self.yolo.getBoxes(self._frame)
            points = self._choose_method(method, i)
            totalPoints, totalLabels = self._postprocessing(points, totalPoints, totalLabels)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in tqdm(range(len(totalPoints))):
            _, self._frame = cap.read()
            points = totalPoints[i]
            labels = totalLabels[i]

            if points.shape[0] != 0:
                u = points[:, 0]
                v = points[:, 1]
                for j in range(u.shape[0]):
                    cv2.circle(self._frame, (u[j], v[j]), 1, self._COLOR_LIST[labels[j]], 4)
            out.write(self._frame)
        cap.release()
        out.release()



class NaiveDrawPoints(DrawPointsBase):
    def __init__(self, datasetPath, videoPath, isCalib=False, fixHeight=False):
        super().__init__(datasetPath, videoPath)
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
            if method == 'img':
                points = self._img_reg(i)

            elif method == 'point':
                points = self._point_reg(i)

            elif method == 'both':
                points = self._point_reg(i)
                points = self._img_reg(i, points)
            else:
                raise ValueError("parameter 'method' error")
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
    def __init__(self, datasetPath, videoPath, modelPath='weights/mlp.pt', device=torch.device("cpu")):
        super().__init__(datasetPath, videoPath)
        self.dataX = None
        self.trueU = None
        self.device = device
        self.modelPath = modelPath
        self.dl = NNDatasetLoader(datasetPath)


    def _choose_method(self, method, i):
        """
        :param method:
            - 'none'
            - 'mlp'
            - 'cnn'
        :param i:
        :return: [n, 3]
        """
        if method == 'mlp':
            points = self._mlp(i)
        elif method == 'cnn':
            points = self._cnn(i)
        elif method == 'none':
            points = self.dl.load2DPoints(i, isCalib=False, fixHeight=False)
        else:
            raise ValueError("parameter 'method' error")
        return points


    def _load_data(self, method):
        if self.dataX is None:
            if method == 'mlp':
                self.dataX, self.trueU = self.dl.loadValid(False, 1, self.device)
            if method == 'cnn':
                self.dataX, self.trueU = self.dl.loadValid(True, 1, self.device)
        return self.dataX, self.trueU


    def _predict(self, i, model):
        x = self.dataX[i]
        u = self.trueU[i]
        z = x[:, 2].cpu().numpy()

        model.eval()
        with torch.no_grad():
            y = model(x)
            v = y[:, 0].cpu().numpy()
        return np.array([u, v, z]).T


    def _mlp(self, i):
        """
        使用 mlp 对点云进行校准
        :param i:
        :return: [m, 3]
        """
        mlp = MLP([self.dl.input_dim, 32, 32, 8, self.dl.output_dim], self.device)
        mlp.load_state_dict(torch.load(self.modelPath))
        return self._predict(i, mlp)


    def _cnn(self, i):
        """
        使用 cnn 对点云进行校准
        :param i:
        :return: [500, 3]
        """
        cnn = CNN(self.dl.input_dim, 500, self.device)
        cnn.load_state_dict(torch.load('cnn.pt'))
        return self._predict(i, cnn)


    def new_video(self, newPath, method):
        self._load_data(method)
        super().new_video(newPath, method)


if __name__ == "__main__":
    original_video_path = './yolov8/runs/detect/track/Video.avi'
    new_video_path = './data/new_point_img_5.avi'

    dp = NNDrawPoints('./data', original_video_path)
    dp.new_video(new_video_path, 'mlp')

    # nimgs = './data/point_img/'
    # video2img(original_video_path, nimgs)
