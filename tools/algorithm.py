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

