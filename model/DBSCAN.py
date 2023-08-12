import torch

from model.union import DisjointSetUnion
from tools.algorithm import square_distance
from tools.stupid_tools import get_point_box_index


class DBSCAN:
    def __init__(self, eps, min_samples):
        self._distance = None
        self.clusterVisit = None
        self.clusterLabel = None
        self.boxLabel = None
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        """
        cluster label 从 1 开始，0 为噪声
        :param X:
        :return:
        """
        cnt = 1
        n = X.shape[0]
        self.clusterVisit = torch.zeros(n)
        self.clusterLabel = torch.zeros(n)
        for i in range(n):
            if self.boxLabel is not None and self.boxLabel[i] == 0: continue
            if self.clusterVisit[i] == 1: continue
            if self._distance is None: self._distance = square_distance(X, X)
            self.clusterVisit[i] = 1
            neighbors = self._region_query(i)
            if len(neighbors) < self.min_samples:
                self.clusterLabel[i] = 0  # 标记为噪声点
            else:
                self.clusterLabel[i] = cnt
                self._expand_cluster(neighbors, cnt)
                cnt += 1
        self._distance = None
        self.clusterLabel = self.clusterLabel.long()
        return self.clusterLabel

    def _region_query(self, i):
        dists = self._distance[i]
        neighbors = torch.nonzero(dists <= self.eps * self.eps).squeeze(dim=1)
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


class ConditionalDBSCAN(DBSCAN):
    INCLUSIVE_THRESHOLD = 0.6
    EXCLUSIVE_THRESHOLD = 0.8

    def __init__(self, eps, min_samples):
        super().__init__(eps, min_samples)

    def conditional_fit(self, X, id, xyxy):
        X = torch.tensor(X, dtype=torch.float32)
        if id is not None:
            n_samples = X.shape[0]
            boxIdIndices = torch.argsort(id) + 1
            self.boxLabel = get_point_box_index(boxIdIndices, xyxy, X)
            self.fit(X)
            clusterLabelUnq, clusterLabelCnt = torch.unique(self.clusterLabel, return_counts=True)
            if clusterLabelUnq[0] == 0: clusterLabelCnt = clusterLabelCnt[1:]
            else: clusterLabelUnq = torch.cat((torch.tensor([0]), clusterLabelUnq), dim=0)

            dsu = DisjointSetUnion(clusterLabelUnq)

            n = clusterLabelUnq[-1]
            m = max(boxIdIndices) + 1
            mat_nm = torch.zeros((n, m), dtype=torch.float32)
            for i in range(n_samples):
                if self.clusterLabel[i] == 0: continue
                cluster = self.clusterLabel[i] - 1
                box = self.boxLabel[i]
                mat_nm[cluster, box] += 1

            for i in range(m):
                prop = mat_nm[:, i] / clusterLabelCnt
                if i == 0: incCluster = torch.nonzero(prop >= ConditionalDBSCAN.EXCLUSIVE_THRESHOLD) + 1
                else:      incCluster = torch.nonzero(prop >= ConditionalDBSCAN.INCLUSIVE_THRESHOLD) + 1
                for j in incCluster: dsu.union(i, j)

            if dsu.unioned:
                for i in range(n_samples):
                    label = dsu.find(self.clusterLabel[i])
                    if label is False:
                        label = self.boxLabel[i]
                    self.clusterLabel[i] = label
                return self.clusterLabel
            else: return self.boxLabel
        return torch.zeros(X.shape[0])
