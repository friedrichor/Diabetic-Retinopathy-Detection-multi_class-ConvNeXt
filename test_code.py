import numpy as np

# .npy文件是numpy专用的二进制文件
arr = np.array([[1, 2], [3, 4]])

# 保存.npy文件
np.save("../arr.npy", arr)
print("save .npy done")

# 读取.npy文件
np.load("../arr.npy")
print(arr)
print("load .npy done")

