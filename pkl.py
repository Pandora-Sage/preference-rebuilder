import os
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder

def create_sparse_matrix(df, user_field, item_field, user_num, item_num):
    """创建用户-物品交互稀疏矩阵（COO格式）"""
    src = df[user_field].values
    tgt = df[item_field].values
    data = np.ones(len(df), dtype=np.float32)
    return sp.coo_matrix((data, (src, tgt)), shape=(user_num, item_num))

def split_dataset(df, splitting_label='x_label'):
    """划分训练/验证/测试集，不过滤新用户"""
    # 按划分标签拆分
    trn_df = df[df[splitting_label] == 0].copy()
    val_df = df[df[splitting_label] == 1].copy()
    tst_df = df[df[splitting_label] == 2].copy()
    
    # 移除划分标签列
    trn_df.drop(splitting_label, inplace=True, axis=1)
    val_df.drop(splitting_label, inplace=True, axis=1)
    tst_df.drop(splitting_label, inplace=True, axis=1)
    
    return trn_df, val_df, tst_df

if __name__ == '__main__':
    # 设置工作路径
    os.chdir('./Datasets/Arts')
    print(f"当前工作目录: {os.getcwd()}")
    
    # 配置参数
    dataset_name = 'Arts'
    user_field = 'userID'       # 用户ID字段
    item_field = 'itemID'       # 物品ID字段
    inter_file = 'Arts-indexed-v4.inter' # 交互数据文件名
    splitting_label = 'x_label' # 划分标签字段
    
    # 读取原始交互数据
    if not os.path.exists(inter_file):
        raise FileNotFoundError(f"交互文件 {inter_file} 不存在于当前目录")
    
    # 读取交互数据（包含userID, itemID, x_label列）
    inter_df = pd.read_csv(inter_file, sep='\t')
    
    # 检查必要字段是否存在
    required_fields = [user_field, item_field, splitting_label]
    missing_fields = [f for f in required_fields if f not in inter_df.columns]
    if missing_fields:
        raise ValueError(f"交互文件缺少必要字段: {missing_fields}")
    
    # 编码用户和物品ID为连续整数
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    inter_df[user_field] = user_encoder.fit_transform(inter_df[user_field])
    inter_df[item_field] = item_encoder.fit_transform(inter_df[item_field])
    
    # 获取用户和物品数量
    user_num = len(user_encoder.classes_)
    item_num = len(item_encoder.classes_)
    print(f"用户数量: {user_num}, 物品数量: {item_num}")
    
    # 计算总交互数目和整体稀疏度
    total_inters = len(inter_df)
    sparsity = 1 - total_inters / (user_num * item_num)
    print(f"总交互数目: {total_inters}")
    print(f"The sparsity of the dataset: {sparsity * 100:.6f}%")
    
    # 划分数据集（不过滤新用户）
    trn_df, val_df, tst_df = split_dataset(inter_df, splitting_label)
    
    # 打印划分后的统计信息
    print("====训练集====")
    print(f"训练集交互数: {len(trn_df)}")
    trn_sparsity = 1 - len(trn_df) / (user_num * item_num)
    print(f"The sparsity of the training set: {trn_sparsity * 100:.6f}%")
    
    print("====验证集====")
    print(f"验证集交互数: {len(val_df)}")
    val_sparsity = 1 - len(val_df) / (user_num * item_num)
    print(f"The sparsity of the validation set: {val_sparsity * 100:.6f}%")
    
    print("====测试集====")
    print(f"测试集交互数: {len(tst_df)}")
    tst_sparsity = 1 - len(tst_df) / (user_num * item_num)
    print(f"The sparsity of the testing set: {tst_sparsity * 100:.6f}%")
    
    # 创建稀疏矩阵
    trn_mat = create_sparse_matrix(trn_df, user_field, item_field, user_num, item_num)
    val_mat = create_sparse_matrix(val_df, user_field, item_field, user_num, item_num)
    tst_mat = create_sparse_matrix(tst_df, user_field, item_field, user_num, item_num)
    
    # 保存为pkl文件
    output_files = {
        0: 'trnMat.pkl',
        1: 'valMat.pkl',
        2: 'tstMat.pkl'
    }
    for idx, mat in [(0, trn_mat), (1, val_mat), (2, tst_mat)]:
        with open(output_files[idx], 'wb') as f:
            pickle.dump(mat, f)
        print(f"已保存 {output_files[idx]}, 形状: {mat.shape}, 非零元素: {mat.nnz}")
