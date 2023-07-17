import pandas as pd
import cv2
import numpy as np


def show_img():
    csv = pd.read_csv(path + filename + ".csv")
    u = csv['u'].tolist()
    v = csv['v'].tolist()
    points = np.array([u, v, np.ones(len(u))])  # [3, n]

    img = cv2.imread(imgPath + filename + ".jpg")
    w = img.shape[0]
    h = img.shape[1]
    da = 0.03
    dx = -30
    dy = 210
    scale = 1
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -np.rad2deg(da), scale)  # [2, 3]

    scaleX = 1
    scaleY = 0.4
    points[0] = points[0] * scaleX
    points[1] = points[1] * scaleY

    points = np.round(np.matmul(m, points)).astype(int).T
    x = points[:, 0] + dx
    y = points[:, 1] + dy

    newX = []
    newY = []
    newIndex = []
    for i in range(len(y)):
        if x[i] > 900:
            newX.append(x[i])
            newY.append(y[i])
            newIndex.append(i)

    da_ = 0
    m_ = cv2.getRotationMatrix2D((w / 2, h / 2), -np.rad2deg(da_), scale)  # [2, 3]
    newPoints = np.array([newX, newY, np.ones(len(newY))])
    newPoints = np.round(np.matmul(m_, newPoints)).astype(int).T
    x_ = newPoints[:, 0]
    y_ = newPoints[:, 1]

    for k, i in enumerate(newIndex):
        x[i] = x_[k] - 0
        y[i] = y_[k] + 10
        if x[i] > 900:
            y[i] -= 0

    for j in range(points.shape[0]):
        cv2.circle(img, (x[j], y[j]), 1, (102, 204, 255), 5)

    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return x, y


def save_label(x, y):
    csv = pd.read_csv(path + filename + ".csv")
    for i in range(csv.shape[0]):
        csv.loc[i, 'u'] = x[i]
        csv.loc[i, 'v'] = y[i]

    csv.to_csv(labelPath + filename + ".csv")


if __name__ == "__main__":
    path = "./data/radar_3/"
    labelPath = "./data/label2/"
    deltaU = 0
    deltaV = 0
    filename = "1669621838.90266"  # down
    imgPath = "./data/point_img/"

    x, y = show_img()
    save_label(x, y)
