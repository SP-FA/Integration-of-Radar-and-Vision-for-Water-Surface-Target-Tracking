import numpy as np
import open3d as o3d
import torch


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
    cnt = 1
    arr = np.array(arr)
    for i in range(len(arr)):
        if arr[i] == 0:
            continue
        if arr[i] not in numDict:
            numDict[arr[i]] = cnt
            arr[i] = cnt
            cnt = (cnt + 1) % COLOR_NUMBER + 1
        else:
            arr[i] = numDict[arr[i]]
    return arr


def get_point_box_index(id, box, points):
    """
    计算每个点分别属于哪个检测框，每个点只属于检测框，conf 较大的框优先
    :param box:
    :param points:
    :return: [n]
    """
    R = in_box(box, points)
    R_index = R.argmax(dim=0)
    R_haveBox = R.sum(dim=0)
    R_union = R_index + R_haveBox
    R_is0 = torch.nonzero(R_union == 0)
    pointBoxId = id[R_index]
    pointBoxId[R_is0] = 0
    return pointBoxId.long()


def in_box(box, points, device=torch.device("cpu")):
    """
    计算每个点分别属于哪个检测框，每个点可能属于多个检测框
    :param box: [m, 4]
    :param points: [n, 3]
    :return: 一个 [m, n] 的布尔矩阵，[i， j] 为 True 说明 点 j 在框 i 中
    """
    m = box.shape[0]
    n = points.shape[0]

    # box = box[:, np.newaxis, :]
    box = box.unsqueeze(1)
    # boxOnes = np.ones((1, n, 4))
    boxOnes = torch.ones((1, n, 4))  # .to(device)
    box = box * boxOnes  # [m, n, 4]

    D2Points = points[:, :2]
    # D2Points = D2Points[:, np.newaxis, :]
    D2Points = D2Points.unsqueeze(1)
    # pointOnes = np.ones((1, m, 2))
    pointOnes = torch.ones((1, m, 2))  # .to(device)
    D2Points = D2Points * pointOnes
    # D2Points = D2Points.transpose(1, 0, 2)  # [m, n, 2]
    D2Points = D2Points.permute(1, 0, 2)

    R = (D2Points[:, :, 0] >= box[:, :, 0]) & (D2Points[:, :, 1] >= box[:, :, 1]) & \
        (D2Points[:, :, 0] <= box[:, :, 2]) & (D2Points[:, :, 1] <= box[:, :, 3])
    return R.long()
