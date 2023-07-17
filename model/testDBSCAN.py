import numpy as np
import matplotlib.pyplot as plt

from draw_point import map_label_to_color
from model.DBSCAN import DBSCAN


def generate_dataset(num_clusters, num_points_per_cluster, noise_ratio):
    # 确定数据集的范围
    x_range = (-10, 10)
    y_range = (-10, 10)

    # 生成聚类中心点
    cluster_centers = []
    for _ in range(num_clusters):
        center_x = np.random.uniform(*x_range)
        center_y = np.random.uniform(*y_range)
        cluster_centers.append((center_x, center_y))

    # 生成数据点
    dataset = []
    for center in cluster_centers:
        points = []
        for _ in range(num_points_per_cluster):
            offset_x = np.random.uniform(-1, 1)
            offset_y = np.random.uniform(-1, 1)
            point_x = center[0] + offset_x
            point_y = center[1] + offset_y
            points.append((point_x, point_y))
        dataset.extend(points)

    # 生成噪声点
    num_noise_points = int(len(dataset) * noise_ratio)
    noise_points = []
    for _ in range(num_noise_points):
        noise_x = np.random.uniform(*x_range)
        noise_y = np.random.uniform(*y_range)
        noise_points.append((noise_x, noise_y))
    dataset.extend(noise_points)

    return np.array(dataset)


colorLst = [
            (1, 0, 0),  # 红色
            (0, 1, 0),  # 绿色
            (0, 0, 1),  # 蓝色
            (1, 1, 0),  # 黄色
            (1, 0, 1),  # 品红色
            (0, 1, 1),  # 青色
            (0.5, 0, 0),  # 深红色
            (0, 0.5, 0),  # 深绿色
            (0, 0, 0.5),  # 深蓝色
            (0.5, 0.5, 0.5),  # 灰色
            (0, 0, 0)  # 白色
        ]


# 生成数据集
num_clusters = 4
num_points_per_cluster = 50
noise_ratio = 0.1
dataset = generate_dataset(num_clusters, num_points_per_cluster, noise_ratio)

# 可视化数据集
# plt.scatter(dataset[:, 0], dataset[:, 1], s=5)
# plt.title('DBSCAN Test Dataset')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()
testSet = np.array([[1798., 469.02816772],
 [1798., 468.77694702],
 [1781., 468.306427  ],
 [1768., 478.890625  ],
 [1768., 476.17279053],
 [1742., 476.50308228],
 [1356., 439.06704712],
 [1347., 438.64151001],
 [1288., 437.3074646 ],
 [1295., 437.28079224],
 [1164., 431.6640625 ],
 [1170., 432.07611084],
 [1179., 430.53881836],
 [1255., 410.14859009],
 [1247., 415.70999146],
 [1256., 417.94189453],
 [1256., 417.67276001],
 [1263., 424.17636108],
 [1263., 427.24307251],
 [1263., 425.19311523],
 [1163., 422.52682495],
 [1170., 421.52468872],
 [1811., 429.49349976],
 [1843., 427.78820801],
 [1780., 431.47540283],
 [1797., 430.5664978 ],
 [1336., 423.55984497],
 [1336., 423.35162354],
 [1287., 405.43508911],
 [1287., 405.18771362],
 [1287., 410.77496338],
 [1287., 412.94259644],
 [1193., 415.44348145],
 [1294., 412.9859314 ],
 [1193., 415.16708374],
 [1555., 396.06536865],
 [1555., 392.67330933],
 [1857., 424.0552063 ],
 [1857., 413.50985718],
 [1767., 435.10974121],
 [1780., 433.08837891],
 [1780., 433.17428589],
 [1399., 430.1008606 ],
 [1194., 417.37973022],
 [1240., 418.43008423],
 [1555., 396.06536865],
 [1555., 393.38061523],
 [1753., 437.2772522 ],
 [1753., 434.5196228 ],
 [1797., 426.33517456],
 [1797., 424.30874634],
 [1811., 420.78451538],
 [1455., 429.76416016],
 [1185., 431.90762329],
 [1194., 430.40097046],
 [1556., 400.34680176],
 [1556., 398.50198364],
 [1767., 451.89865112],
 [1767., 453.16378784],
 [1767., 454.53759766],
 [1797., 437.90435791],
 [1797., 432.08908081],
 [1797., 432.17556763],
 [1484., 442.31765747],
 [1194., 422.09506226],
 [1185., 421.56604004],
 [1698., 406.31512451],
 [1545., 405.95562744],
 [1545., 405.58322144],
 [ 949., 404.58596802],
 [ 937., 404.35745239],
 [ 931., 400.41885376],
 [ 916., 399.78128052],
 [1811., 437.07116699],
 [1812., 470.90408325],
 [1546., 445.86581421],
 [1813., 473.10232544],
 [1445., 447.83010864],
 [1337., 444.37161255],
 [1256., 441.45028687],
 [1232., 440.68502808],
 [1241., 437.6940918 ],
 [1179., 434.06744385],
 [1155., 432.61120605],
 [1272., 438.09463501],
 [1271., 419.31777954],
 [1271., 416.81478882],
 [1271., 418.7835083 ],
 [1278., 422.54037476],
 [1279., 424.70184326],
 [1278., 421.99212646],
 [1185., 428.49465942],
 [1185., 425.08432007]])

eps = 50
min_samples = 4
dbscan = DBSCAN(eps, min_samples)
labels = dbscan.fit(testSet)
labels = map_label_to_color(labels)

plt.axis([0, 1920, 0, 1080])
plt.scatter(testSet[:, 0], testSet[:, 1], s=5, c=[colorLst[i] for i in labels])
plt.title('DBSCAN Test Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
