import time
import numpy as np
from tqdm import tqdm
from tools.algorithm import square_distance
import torch


Xlst = []
epoch = 10

for i in range(epoch):
    X = torch.rand(10000, 3).cuda()
    Y = torch.rand(10000, 3).cuda()
    Xlst.append([X, Y])

# 记录程序开始时间
start_time = time.time()

# 在这里运行你的Python程序代码

for i in tqdm(Xlst):
    square_distance(i[0], i[1])

# 记录程序结束时间
end_time = time.time()

# 计算程序运行时间
run_time = end_time - start_time

print("程序运行时间：", run_time, "秒")
