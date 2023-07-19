import numpy as np


CAMERA_HEIGHT = 0.77


def project_pcl_to_image(DatasetLoader, i, fixHeight):
    """
    Project the point cloud to the image
    :param DatasetLoader:
    :param i:
    :param fixHeight: 是否使用 fix height 矫正
    :return: [n, 2], ndarray
    """
    points = DatasetLoader.load3DPoints(i)
    compH = DatasetLoader.loadCompH(i)
    pitch = DatasetLoader.loadPitch(i)
    extrinsic, intrinsic = DatasetLoader.loadMatrix(i)

    if fixHeight is True:
        points = fix_height_calibration(points, compH, pitch)

    location = np.hstack((points, np.ones((points.shape[0], 1))))  # [n, 3] cat [n, 1] = [n, 4]
    radar_points_camera_frame = extrinsic.dot(location.T).T  # [4, 4] @ [4, n]; [n, 4]

    uvs = project_3d_to_2d(radar_points_camera_frame, intrinsic)
    return uvs[:, :2].astype(int)


def project_3d_to_2d(points, intrinsic):
    """
    :param points: 经过外参矫正后的 [n, 4] 点云
    :param intrinsic: 内参矩阵 [3, 4]
    :return: [n, 3]
    """
    uvw = intrinsic.dot(points.T)  # [3, 4] @ [4, n] = [3, n]
    uvw /= uvw[2]
    uvs = np.round(uvw.T).astype(int)
    return uvs  #


def fix_height_calibration(points, compH, pitch):
    pitch = np.deg2rad(pitch)
    x = points[:, 0]
    z = points[:, 2]

    H = abs(compH - CAMERA_HEIGHT)
    new_y = - (H / np.cos(pitch) + z * np.tan(pitch))
    return np.array([x, new_y, z]).T
