import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from model.kdTree import KDTree
from tools.visualization import draw_plots

csvPath = "./data/radar_3"
files = os.listdir(csvPath)

totDistLst = []
for i in tqdm(files):
    df = pd.read_csv(os.path.join(csvPath, i))
    points = np.array([df['u'].tolist(), df['v'].tolist(), df['z'].tolist()]).T
    kdt = KDTree(points, 4)

    for j in range(points.shape[0]):
        points, dist = kdt.search_nearest(j, isInner=True)
        totDist = -dist[0]
        totDistLst.append(totDist)

totDistLst = sorted(totDistLst, reverse=True)
totDistLst = totDistLst[:10000]
draw_plots("k-distance", totDistLst, totDistLst)
