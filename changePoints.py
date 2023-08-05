import pandas as pd
import cv2
import numpy as np

from util.load_data import DatasetLoader


def show_img(datasetPath=None, i=None):
    csv = pd.read_csv(path + str(float(filename)) + ".csv")
    try:
        u = csv['u'].tolist()
        v = csv['v'].tolist()
        points = np.array([u, v, np.ones(len(u))])  # [3, n]
    except KeyError:
        dl = DatasetLoader(datasetPath)
        points = dl.load2DPoints(i-1, True, False)
        points = np.hstack((points, np.ones(len(points)).reshape(-1, 1))).T

    img = cv2.imread(imgPath + filename + ".jpg")

    w = img.shape[0]
    h = img.shape[1]
    da = 0
    dx = 0
    dy = -30
    scale = 1
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -np.rad2deg(da), scale)  # [2, 3]

    scaleX = 1
    scaleY = 1
    points[0] = points[0] * scaleX
    points[1] = points[1] * scaleY

    points = np.round(np.matmul(m, points)).astype(int).T
    x = points[:, 0] + dx
    y = points[:, 1] + dy

    newX = []
    newY = []
    newIndex = []
    for i in range(len(y)):
        # if y[i] > 570:
        #     y[i] += -220
        if x[i] > 850:
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
        x[i] = x_[k] + 0
        y[i] = y_[k] + 30
        # if y[i] > 570:
        #     y[i] += 0
        # elif x[i] > 1120:
        #     y[i] += 20
        # if x[i] < 860 and y[i] > 430:
        #     y[i] += -90
        # if x[i] < 900: y[i] += 10
        # if x[i] < 700: y[i] += 20
        # if x[i] > 750: y[i] -= 15
        # if y[i] > 420: y[i] -= 20
        # if y[i] > 460: y[i] -= 80

    # for i in range(len(y)):
    #     if y[i] > 710:
    #         y[i] += -220
    #
    # for i in range(len(y)):
    #     if y[i] > 555 and x[i] > 590:
    #         y[i] += -100
    #
    # for i in range(len(y)):
    #     if x[i] < 850:
    #         y[i] += -30
    #
    # for i in range(len(y)):
    #     if x[i] < 600:
    #         y[i] += -20
    #
    # for i in range(len(y)):
    #     if x[i] < 866:
    #         y[i] += -20

    for j in range(points.shape[0]):
        cv2.circle(img, (x[j], y[j]), 1, (102, 204, 255), 5)

    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return x, y


def save_label(x, y):
    csv = pd.read_csv(path + str(float(filename)) + ".csv")
    for i in range(csv.shape[0]):
        csv.loc[i, 'u'] = x[i]
        csv.loc[i, 'v'] = y[i]

    csv.to_csv(labelPath + filename + ".csv")


if __name__ == "__main__":
    datasetPath = r"H:\dataset\Tracking\YOLO_timestamp\44"
    path = r"H:\dataset\Tracking\YOLO_timestamp\44/radar/"
    labelPath = r"H:\dataset\Tracking\YOLO_timestamp\44\point_label/"
    deltaU = 0
    deltaV = 0
    filename = "1669621580.75920"  # up
    imgPath = r"H:\dataset\Tracking\YOLO_timestamp\44\images/"

    x, y = show_img(datasetPath, i=296)
    save_label(x, y)
