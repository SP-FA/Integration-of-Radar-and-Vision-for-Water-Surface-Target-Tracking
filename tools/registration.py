import numpy as np
import cv2 as cv
import open3d as o3d

from tools.algorithm import affine


def image_registration(points, frame, lastFrame):
    """
    :param points: [n, 2]
    """
    H, _ = _img_reg(frame, lastFrame)
    u, v = points[:, 0], points[:, 1]
    points = affine(u, v, H)
    return points, H


def point_cloud_registration(source, target):
    """
    :param source: [n, 2]
    :param target: [n, 2]
    :return:
    """
    source = np.concatenate((source, np.zeros(source.shape[0])), axis=1).T
    target = np.concatenate((target, np.zeros(target.shape[0])), axis=1).T
    points = _point_reg(source, target)
    return points


def _img_reg(img1, img2, id=None):
    # 初始化 AKAZE 探测器
    akaze = cv.AKAZE_create()
    # 使用 SIFT 查找关键点和描述
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    # BFMatcher 默认参数
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # 旋转测试
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    # 选择匹配关键点
    ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # 计算 homography
    H, status = cv.findHomography(ref_matched_kpts, sensed_matched_kpts, cv.RANSAC, 5.0)
    # 变换
    warped_image = cv.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))
    if id is None:
        return H, warped_image
    else:
        cv.imwrite('./data/new_3/warped%d.jpg' % id, warped_image)


def _point_reg(source, target):
    trans_init = np.asarray([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, 0.02, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )

    source.transform(reg_p2p.transformation)
    return np.asarray(source.points)[:, :2].astype(int)

