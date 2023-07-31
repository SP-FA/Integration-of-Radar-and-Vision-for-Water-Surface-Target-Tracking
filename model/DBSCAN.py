from collections import Counter
import numpy as np
import torch

from model.yoloIter import YOLOIterator
from tools.algorithm import square_distance
from tools.stupid_tools import in_box, get_point_box_index


class DBSCAN:
    def __init__(self, eps, min_samples):
        self._distance = None
        self.clusterVisit = None
        self.clusterLabel = None
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        """
        cluster label 从 1 开始，0 为噪声
        :param X:
        :return:
        """
        self._distance = square_distance(X, X)

        n = X.shape[0]
        # self.clusterVisit = np.zeros(n)
        # self.clusterLabel = np.zeros(n)
        self.clusterVisit = torch.zeros(n)
        self.clusterLabel = torch.zeros(n)

        cnt = 1
        for i in range(n):
            if self.clusterVisit[i] == 1: continue
            self.clusterVisit[i] = 1
            neighbors = self._region_query(i)
            if len(neighbors) < self.min_samples:
                self.clusterLabel[i] = 0  # 标记为噪声点
            else:
                self.clusterLabel[i] = cnt
                self._expand_cluster(neighbors, cnt)
                cnt += 1
        self.clusterLabel = self.clusterLabel.long()
        return self.clusterLabel


    def _region_query(self, i):
        dists = self._distance[i]
        neighbors = torch.nonzero(dists <= self.eps).squeeze(dim=1)
        return neighbors


    def _expand_cluster(self, neighbors, cnt):
        j = 0
        while j < len(neighbors):
            neighbor = neighbors[j]
            if self.clusterVisit[neighbor] == 0:
                self.clusterVisit[neighbor] = 1
                self.clusterLabel[neighbor] = cnt
                newNeighbors = self._region_query(neighbor)
                if len(newNeighbors) >= self.min_samples:
                    neighbors = torch.concatenate((neighbors, newNeighbors), dim=0)
            j += 1


class VisionDBSCAN(DBSCAN):
    INCLUSIVE_THRESHOLD = 0.7
    EXCLUSIVE_THRESHOLD = 0.8
    cntt = 0

    def __init__(self, eps, min_samples):
        super().__init__(eps, min_samples)
        self.boxLabel = None
        self.xyxy = None
        self.yolo = YOLOIterator("../weights/best.pt")

    def vision_fit(self, X, frame):
        self.cntt += 1
        X = torch.tensor(X)
        self.fit(X)
        xyxy, cls, id, conf = self.yolo.getBoxes(frame)
        if id is not None:
            # self.xyxy = xyxy.astype(int)
            self.xyxy = xyxy.to(torch.int)
            n_samples = X.shape[0]
            # boxVisit = torch.zeros(n_samples)
            # boxLabel = torch.zeros(n_samples)
            # fullBoxId = torch.cat((torch.zeros(1), id))
            boxIdIndices = torch.argsort(id) + 1
            # boxVisit = np.zeros(n_samples)
            # boxLabel = np.zeros(n_samples)
            # fullBoxId = np.concatenate((np.zeros(1), id))
            # R = in_box(self.xyxy, X)
            # for i in range(R.shape[0]):
            #     for j in range(R.shape[1]):
            #         if boxVisit[j] == 1: continue
            #         if R[i, j]:
            #             boxLabel[j] = id[i]
            #             boxVisit[j] = 1

            self.boxLabel = get_point_box_index(boxIdIndices, self.xyxy, X)

            # self.boxLabel = boxLabel.astype(int)
            # clusterLabelUnq = Counter(self.clusterLabel)
            # clusterLabelUnq = {int(key): float(value) for key, value in clusterLabelUnq.items()}
            clusterLabelUnq, clusterLabelCnt = torch.unique(self.clusterLabel, return_counts=True)
            if clusterLabelUnq[0] != 0: clusterLabelCnt = torch.cat((torch.zeros(1), clusterLabelCnt))

            n = clusterLabelUnq[-1] + 1
            m = max(boxIdIndices) + 1

            # mat_nm = np.zeros((n, m))
            mat_nm = torch.zeros((n, m), dtype=torch.float32)
            for i in range(n_samples):
                clusterId = self.clusterLabel[i]
                boxId = self.boxLabel[i]
                # mat_nm[clusterId, np.where(fullBoxId == boxId)] += 1
                mat_nm[clusterId, boxId] += 1

            inclusiveLst = []
            for i in range(m):
                prop = mat_nm[:, i] / clusterLabelCnt
                prop = prop[1:]
                # if self.cntt >= 91:
                #     print(prop)
                if i == 0: incCluster = torch.nonzero((prop != 1) & (prop >= VisionDBSCAN.EXCLUSIVE_THRESHOLD)) + 1
                else:      incCluster = torch.nonzero((prop != 1) & (prop >= VisionDBSCAN.INCLUSIVE_THRESHOLD)) + 1

                for j in incCluster: inclusiveLst.append([j, i])

                # for j in range(m):
                    # prop = mat_nm[i][j] / clusterLabelCnt[i]
                    # if j == 0 \
                    #         and prop != 1 \
                    #         and prop >= VisionDBSCAN.EXCLUSIVE_THRESHOLD:
                    #     inclusiveLst.append([i, 0])
                    #     break
                    # elif j != 0 \
                    #         and prop != 1 \
                    #         and prop >= VisionDBSCAN.INCLUSIVE_THRESHOLD:
                    #     inclusiveLst.append([i, fullBoxId[j]])
                    #     break

            # idWith0 = torch.cat((torch.zeros(1), id))
            if inclusiveLst:
                finalLabel = torch.empty(self.boxLabel.shape[0])
                # inclusiveLst = np.array(inclusiveLst).T
                inclusiveLst = torch.tensor(inclusiveLst).permute(1, 0)
                for i in range(n_samples):
                    clusterId = self.clusterLabel[i]
                    if clusterId in inclusiveLst[0]:
                        # self.clusterLabel[i] = inclusiveLst[1, np.where(inclusiveLst[0] == clusterId)]
                        self.clusterLabel[i] = inclusiveLst[1, torch.nonzero(inclusiveLst[0] == clusterId)]
                    else:
                        self.clusterLabel[i] = self.boxLabel[i]
                # for i in clusterLabelUnq:
                #     print(finalLabel[self.clusterLabel == i], self.boxLabel[self.clusterLabel == i])
                #     if i in inclusiveLst[0]:
                #         finalLabel[self.clusterLabel == i] = inclusiveLst[1, torch.nonzero(inclusiveLst[0] == i)]
                #     else: finalLabel[self.clusterLabel == i] = self.boxLabel[self.clusterLabel == i]
            else: return self.boxLabel
        # return self.clusterLabel.astype(int)
        return self.clusterLabel.long()

