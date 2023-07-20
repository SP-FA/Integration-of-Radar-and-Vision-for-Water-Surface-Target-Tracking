from collections import Counter
import numpy as np
import torch

from model.yoloIter import YOLOIterator
from tools.stupid_tools import in_box


class DBSCAN:
    def __init__(self, eps, min_samples):
        self.distance = None
        self.visit = None
        self.label = None
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        """
        分类 id 从 1 开始，0 为噪声
        :param X:
        :return:
        """
        self._distance(X)

        n_samples = X.shape[0]
        self.visit = np.zeros(n_samples)
        self.label = np.zeros(n_samples)

        clusterId = 1
        for i in range(n_samples):
            if self.visit[i] == 1:
                continue
            self.visit[i] = 1

            neighbors = self._region_query(i)
            if len(neighbors) < self.min_samples:
                self.label[i] = 0  # 标记为噪声点
            else:
                self.label[i] = clusterId
                self._expand_cluster(neighbors, clusterId)
                clusterId += 1
        self.label = self.label.astype(int)
        return self.label


    def _distance(self, X):
        self.distance = np.linalg.norm(X[:, np.newaxis] - X, axis=-1)


    def _region_query(self, i):
        dists = self.distance[i]
        neighbors = np.argwhere(dists <= self.eps).squeeze(axis=1)
        return neighbors


    def _expand_cluster(self, neighbors, clusterId):
        j = 0
        while j < len(neighbors):
            neighbor = neighbors[j]
            if self.visit[neighbor] == 0:
                self.visit[neighbor] = 1
                self.label[neighbor] = clusterId
                new_neighbors = self._region_query(neighbor)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate((neighbors, new_neighbors), axis=0)
            j += 1


class VisionDBSCAN(DBSCAN):
    INCLUSIVE_THRESHOLD = 0.7
    EXCLUSIVE_THRESHOLD = 0.9

    def __init__(self, eps, min_samples):
        super().__init__(eps, min_samples)
        self.xyxy = None
        self.yolo = YOLOIterator("../weights/best.pt")

    def vision_fit(self, X, frame):
        self.fit(X)
        self.xyxy, cls, id, conf = self.yolo.getBoxes(frame)
        if id is not None:
            n_samples = X.shape[0]
            boxVisit = np.zeros(n_samples)
            boxLabel = np.zeros(n_samples)
            fullBoxId = np.concatenate((np.zeros(1), id))
            self._distance(X)
            R = in_box(self.xyxy, X)
            for i in range(R.shape[0]):
                for j in range(R.shape[1]):
                    if boxVisit[j] == 1: continue
                    if R[i, j]:
                        boxLabel[j] = id[i]
                        boxVisit[j] = 1

            boxLabel = boxLabel.astype(int)
            labelDict = Counter(self.label)
            labelDict = {int(key): float(value) for key, value in labelDict.items()}

            n = max(labelDict) + 1
            m = len(fullBoxId)

            mat_nm = np.zeros((n, m))
            for i in range(n_samples):
                clusterId = self.label[i]
                boxId = boxLabel[i]
                mat_nm[clusterId, np.where(fullBoxId == boxId)] += 1

            inclusiveLst = []
            for i in labelDict.keys():
                if i == 0: continue
                for j in range(len(fullBoxId)):
                    prop = mat_nm[i][j] / labelDict[i]
                    if j == 0 \
                            and prop != 1 \
                            and prop >= VisionDBSCAN.EXCLUSIVE_THRESHOLD:
                        inclusiveLst.append([i, fullBoxId[j]])
                        break
                    elif j != 0 \
                            and prop != 1 \
                            and prop >= VisionDBSCAN.INCLUSIVE_THRESHOLD:
                        inclusiveLst.append([i, fullBoxId[j]])
                        break

            if inclusiveLst:
                inclusiveLst = np.array(inclusiveLst).T
                for i in range(n_samples):
                    clusterId = self.label[i]
                    if clusterId in inclusiveLst[0]:
                        self.label[i] = inclusiveLst[1, np.where(inclusiveLst[0] == clusterId)]
                    else:
                        self.label[i] = boxLabel[i]
            else:
                self.label = boxLabel
        return self.label.astype(int)


# class torch_DBSCAN:
#     def __init__(self, eps, min_samples, device=torch.device("cpu")):
#         self.distance = None
#         self.visited = None
#         self.labels = None
#         self.eps = eps
#         self.min_samples = min_samples
#         self.device = device
#
#
#     def fit(self, X):
#         X = X.to(self.device)
#         self._distance(X)
#
#         n_samples = X.shape[0]
#         self.visited = torch.zeros(n_samples).to(self.device)
#         self.labels = torch.zeros(n_samples).to(self.device)
#
#         clusterId = 1
#         for i in range(n_samples):
#             if self.visited[i] == 1:
#                 continue
#             self.visited[i] = 1
#
#             neighbors = self._region_query(i)
#             if len(neighbors) < self.min_samples:
#                 self.labels[i] = -1  # 标记为噪声点
#             else:
#                 self._expand_cluster(i, neighbors, clusterId)
#                 clusterId += 1
#
#         return self.labels
#
#
#     def _distance(self, X):
#         differences = X.unsqueeze(1) - X.unsqueeze(0)
#         squaredDist = torch.sum(differences ** 2, dim=2)
#         self.distance = torch.sqrt(squaredDist)
#
#
#     def _region_query(self, i):
#         dists = self.distance[i, :]
#         neighbors = torch.nonzero(dists <= self.eps).squeeze(1)
#         return neighbors
#
#
#     def _expand_cluster(self, i, neighbors, cluster_id):
#         self.labels[i] = cluster_id
#         j = 0
#         while j < len(neighbors):
#             neighbor = neighbors[j]
#             if self.visited[neighbor] == 0:
#                 self.visited[neighbor] = 1
#                 new_neighbors = self._region_query(neighbor)
#                 if len(new_neighbors) >= self.min_samples:
#                     neighbors = torch.cat((neighbors, new_neighbors))
#             if self.labels[neighbor] == 0:
#                 self.labels[neighbor] = cluster_id
#             j += 1
