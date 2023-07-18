import numpy as np
import open3d as o3d


def remove_outlier(points):
    """
    :param points: [n, 3], ndarray
    :return: an o3d 3D point cloud object
    """
    if points.shape[-1] != 3:
        points = points.T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    return pcd


def map_label_to_color(arr):
    COLOR_NUMBER = 10
    numDict = {}
    cnt = 0
    for i in range(len(arr)):
        if arr[i] == -1:
            continue
        if arr[i] not in numDict:
            numDict[arr[i]] = cnt
            arr[i] = cnt
            cnt = (cnt + 1) % COLOR_NUMBER
        else:
            arr[i] = numDict[arr[i]]
    return arr


def in_box(box, points):
    """

    :param box: [m, 4]
    :param points: [n, 3]
    :return: 一个 [m, n] 的布尔矩阵，[i， j] 为 True 说明 点 j 在框 i 中
    """
    m = box.shape[0]
    n = points.shape[0]

    box = box[:, np.newaxis, :]
    boxOnes = np.ones((1, n, 4))
    box = box * boxOnes  # [m, n, 4]

    D2Points = points[:, :2]
    D2Points = D2Points[:, np.newaxis, :]
    pointOnes = np.ones((1, m, 2))
    D2Points = D2Points * pointOnes
    D2Points = D2Points.transpose(1, 0, 2)  # [m, n, 2]

    R = (D2Points[:, :, 0] >= box[:, :, 0]) & \
        (D2Points[:, :, 1] >= box[:, :, 1]) & \
        (D2Points[:, :, 0] <= box[:, :, 2]) & \
        (D2Points[:, :, 1] <= box[:, :, 3])
    return R

