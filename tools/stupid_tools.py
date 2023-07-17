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
