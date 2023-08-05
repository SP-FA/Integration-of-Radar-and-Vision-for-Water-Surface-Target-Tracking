import numpy as np
import torch


def kalman(z, x_last=0, p_last=0, Q=0.1, R=8):
    """
    一维卡尔曼滤波算法
    :param z: 测量值
    :param x_last: 上一帧的 x 值
    :param p_last: 上一帧的协方差矩阵
    :param Q: 过程噪声协方差
    :param R:观测噪声
    :return:
    """
    x_mid = x_last
    p_mid = p_last + Q
    kg = p_mid / (p_mid + R)
    x_now = x_mid + kg * (z - x_mid)
    p_now = (1 - kg) * p_mid
    p_last = p_now
    return x_now, p_last


def affine(x, y, H):
    od = np.array([x, y, np.ones(x.shape[0])])  # [3, n]
    transformed = np.matmul(H, od)
    if H.shape[0] == 3:
        transformed /= transformed[2]
    transformed = np.round(transformed).astype(int)
    return transformed


def square_distance(X1, X2):
    """
    Calculate Euclid distance between each two points.
    X1^T * X2 = xn * xm + yn * ym + zn * zm；
    sum1 = sum(X1^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum2 = sum(X2^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum1 + sum2 - 2 * X1^T * X2
    Input:
        X1: torch.tensor, [B, N, C]
        X2: torch.tensor, [B, M, C]
    Output:
        dist: torch.tensor.cuda(), [B, N, M]
    """
    if torch.cuda.is_available():
        X1 = X1.cuda()
        X2 = X2.cuda()

    if len(X1.shape) == 2: X1 = X1.unsqueeze(dim=0)
    if len(X2.shape) == 2: X2 = X2.unsqueeze(dim=0)

    B, N, _ = X1.shape
    _, M, _ = X2.shape

    dist = -2 * torch.matmul(X1, X2.permute(0, 2, 1))
    dist += torch.sum(X1 ** 2, -1).view(B, N, 1)
    dist += torch.sum(X2 ** 2, -1).view(B, 1, M)
    if dist.shape[0] == 1: dist = dist.squeeze(0)
    return dist


def feature_trans_regularizer(trans):
    d = trans.size()[1]
    batchSize = trans.size()[0]
    I = torch.eye(d).repeat(batchSize, 1, 1)
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I))
    return loss
