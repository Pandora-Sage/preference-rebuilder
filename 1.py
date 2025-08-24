import numpy as np

# 载入特征文件
text_feats  = np.load("./Datasets/clothing/text_feat.npy")
image_feats = np.load("./Datasets/clothing/image_feat.npy")

# 打印形状
print("文本特征形状:", text_feats.shape)
print("图像特征形状:", image_feats.shape)


import pickle
import scipy.sparse as sp

# 替换为你的实际文件路径
pkl_files = [
    "./Datasets/clothing/trnMat.pkl",
    "./Datasets/clothing/valMat.pkl",
    "./Datasets/clothing/tstMat.pkl",
]

for fp in pkl_files:
    # 1. 加载稀疏矩阵
    with open(fp, "rb") as f:
        mat = pickle.load(f)  # 应为 scipy.sparse.coo_matrix 或其他稀疏类型
    
    # 2. 获取基本信息
    num_users, num_items = mat.shape
    num_interactions = mat.nnz  # 非零元素数
    
    # 3. 打印结果
    print(f"文件: {fp}")
    print(f"  用户数      : {num_users}")
    print(f"  物品数      : {num_items}")
    print(f"  交互条数    : {num_interactions}")
    print("-" * 40)

