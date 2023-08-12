import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from model.kdTree import KDTree
from tools.visualization import draw_plots

csvPath = "./data/radar"
files = os.listdir(csvPath)

totDistLst = []
for i in tqdm(files):
    df = pd.read_csv(os.path.join(csvPath, i))
    # points = np.array([df['u'].tolist(), df['v'].tolist(), df['z'].tolist(),
    #                    df["rcs"].tolist(), df["doppler"].tolist()]).T
    points = np.array([df['u'].tolist(), df['v'].tolist(), df['z'].tolist()]).T
    kdt = KDTree(points, 4)

    for j in range(points.shape[0]):
        points, dist = kdt.search_nearest(j, isInner=True)
        totDist = -dist[0]
        totDistLst.append(totDist)

totDistLst = sorted(totDistLst)
# totDistLst = totDistLst[500000:]
# print(totDistLst)
draw_plots("k-distance", totDistLst, totDistLst)
