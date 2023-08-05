import heapq
import numpy as np


class KDTree:
    class KDNode:
        def __init__(self, point, left=None, right=None, split_dim=None):
            self.point = point
            self.left = left
            self.right = right
            self.splitDim = split_dim

        def __lt__(self, other):
            for i in range(len(self.point)):
                if self.point[i] == other.point[i]:
                    continue
                return self.point[i] < other.point[i]
            return self.point[-1] < other.point[-1]

    def __init__(self, points, k=4):
        self.points = points
        self.root = None
        self.K = k
        self._disMat = None
        self._build()

    def _distance(self, point1, point2):
        return np.sum((point2 - point1) ** 2)

    def _build(self, depth=0, leftId=None, rightId=None):
        if depth == 0:
            leftId = 0
            rightId = self.points.shape[0] - 1

        if leftId > rightId: return None

        ndims = self.points.shape[1]
        splitDim = depth % ndims

        sortedIndexes = np.argsort(self.points[:, splitDim])
        points = self.points[sortedIndexes]

        mid = (leftId + rightId) // 2
        median = points[mid]

        left = self._build(depth + 1, leftId, mid - 1)
        right = self._build(depth + 1, mid + 1, rightId)

        kdn = self.KDNode(median, left, right, splitDim)
        if depth == 0: self.root = kdn
        return kdn


    def _search(self, node, target, heap):
        if node is None: return
        dist = self._distance(node.point, target)
        heapq.heappush(heap, (-dist, node))

        if len(heap) > self.K: heapq.heappop(heap)

        splitDim = node.splitDim
        targetVal = target[splitDim]
        nodeVal = node.point[splitDim]

        if targetVal < nodeVal:
            self._search(node.left, target, heap)
            if targetVal + abs(heap[0][0]) >= nodeVal:
                self._search(node.right, target, heap)
        else:
            self._search(node.right, target, heap)
            if targetVal - abs(heap[0][0]) <= nodeVal:
                self._search(node.left, target, heap)

    def search_nearest(self, target):
        heap = []
        self._search(self.root, target, heap)
        points = [i.point for _, i in heap]
        dists = [d for d, _ in heap]
        return points, dists
