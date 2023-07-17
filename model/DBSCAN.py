import numpy as np
import torch


class DBSCAN:
    def __init__(self, eps, min_samples):
        self.distance = None
        self.visited = None
        self.labels = None
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        self._distance(X)

        n_samples = X.shape[0]
        self.visited = np.zeros(n_samples)
        self.labels = np.zeros(n_samples)

        clusterId = 1
        for i in range(n_samples):
            if self.visited[i] == 1:
                continue
            self.visited[i] = 1

            neighbors = self._region_query(i)
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1  # 标记为噪声点
            else:
                self._expand_cluster(i, neighbors, clusterId)
                clusterId += 1

        return self.labels.astype(int)


    def _distance(self, X):
        self.distance = np.linalg.norm(X[:, np.newaxis] - X, axis=-1)
        # distance = self.distance.reshape(-1, 1)
        # distance = sorted(distance)
        # draw_plots("distance", distance, distance)


    def _region_query(self, i):
        dists = self.distance[i]
        neighbors = np.argwhere(dists <= self.eps).squeeze(axis=1)
        return neighbors


    def _expand_cluster(self, i, neighbors, cluster_id):
        self.labels[i] = cluster_id
        j = 0
        while j < len(neighbors):
            neighbor = neighbors[j]
            if self.visited[neighbor] == 0:
                self.visited[neighbor] = 1
                new_neighbors = self._region_query(neighbor)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate((neighbors, new_neighbors), axis=0)
            if self.labels[neighbor] == 0:
                self.labels[neighbor] = cluster_id
            j += 1


class torch_DBSCAN:
    def __init__(self, eps, min_samples, device=torch.device("cpu")):
        self.distance = None
        self.visited = None
        self.labels = None
        self.eps = eps
        self.min_samples = min_samples
        self.device = device

    def fit(self, X):
        X = X.to(self.device)
        self._distance(X)

        n_samples = X.shape[0]
        self.visited = torch.zeros(n_samples).to(self.device)
        self.labels = torch.zeros(n_samples).to(self.device)

        clusterId = 1
        for i in range(n_samples):
            if self.visited[i] == 1:
                continue
            self.visited[i] = 1

            neighbors = self._region_query(i)
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1  # 标记为噪声点
            else:
                self._expand_cluster(i, neighbors, clusterId)
                clusterId += 1

        return self.labels


    def _distance(self, X):
        differences = X.unsqueeze(1) - X.unsqueeze(0)
        squaredDist = torch.sum(differences ** 2, dim=2)
        self.distance = torch.sqrt(squaredDist)


    def _region_query(self, i):
        dists = self.distance[i, :]
        neighbors = torch.nonzero(dists <= self.eps).squeeze(1)
        return neighbors


    def _expand_cluster(self, i, neighbors, cluster_id):
        self.labels[i] = cluster_id
        j = 0
        while j < len(neighbors):
            neighbor = neighbors[j]
            if self.visited[neighbor] == 0:
                self.visited[neighbor] = 1
                new_neighbors = self._region_query(neighbor)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = torch.cat((neighbors, new_neighbors))
            if self.labels[neighbor] == 0:
                self.labels[neighbor] = cluster_id
            j += 1
