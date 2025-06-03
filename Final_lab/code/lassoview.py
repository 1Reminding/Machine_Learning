import numpy as np
import scanpy as sc
import pandas as pd
import random
from collections import Counter
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import h5py
import scipy.sparse
import time
import random
import numpy as np
import pkg_resources
import sys
import os
import anndata as ad
import label_propagation as LPA

def LPARefine(adata, selected, function="anndata", use_model=LabelPropagation, do_correct=True):
    """ 
    使用标签传播算法进行细胞类型预测的函数
    参数:
    adata: AnnData 对象或 h5ad 文件路径
    selected: 用户选择的细胞索引列表
    function: 指定数据源类型，"anndata" 或 "h5adfile"
    use_model: 用于标签传播的模型（默认为 LabelPropagation）
    do_correct: 是否纠正预测结果（默认为 True）
    返回:
    经过预测的细胞索引列表
    """
    
    # 初始化连接性矩阵
    mat = 1
    if function == "anndata":
        # 从 AnnData 对象中获取连接性矩阵
        mat = adata.obsp['connectivities']
        if not scipy.sparse.issparse(mat):
            mat = scipy.sparse.csr_matrix(mat)
    elif function == "h5adfile":
        # 从 h5ad 文件中读取连接性矩阵
        with h5py.File(adata, 'r') as f:
            group = f['obsp']['connectivities']
            data = group['data'][:]
            indices = group['indices'][:]
            indptr = group['indptr'][:]
            shape = (group.attrs['shape'][0], group.attrs['shape'][1])
            mat = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
    else:
        print('无效的 function 参数，应为 "anndata" 或 "h5adfile"')
        return

    # 将连接性矩阵转换为 COO 格式以便处理
    coo = mat.tocoo()
    rows, cols, data = coo.row, coo.col, coo.data

    # 获取细胞标签
    if function == "anndata":
        obs_col = 'annotation'
        if obs_col not in adata.obs:
            obs_col = 'leiden-1'  # 使用备用列名

        # 获取细胞的标签
        if "codes" in adata.obs[obs_col]:
            mat = adata.obs[obs_col]['codes'].values
        else:
            mat = adata.obs[obs_col].values
    elif function == "h5adfile":
        with h5py.File(adata, 'r') as h5file:
            obs_group = h5file['obs']
            obs_col = 'annotation'
            if obs_col not in obs_group:
                obs_col = 'leiden-1'

            if "codes" in obs_group[obs_col]:
                mat = obs_group[obs_col]['codes'][:]
            else:
                mat = obs_group[obs_col][:]
    else:
        print('无效的 function 参数，应为 "anndata" 或 "h5adfile"')
        return

    # 创建类别映射
    val = {}
    for i in np.unique(mat):
        val[i] = len(val)
    val[len(val)] = len(val)  # 为未分类的项分配一个额外的类别
    print("val: ", val)

    # 创建标签传播所需的稀疏矩阵
    X = LPA.matCoo(mat.shape[0], mat.shape[0])  # 假设 LPA.matCoo 是一个创建 COO 矩阵的函数
    for i in range(len(data)):
        X.append(rows[i], cols[i], data[i])

    # 初始化标签
    y_label = LPA.mat(mat.shape[0], len(val))  # 假设 LPA.mat 是一个初始化矩阵的函数
    random_list = random.sample(range(mat.shape[0]), int(mat.shape[0] * 0.1))  # 随机选择 10% 的细胞
    select_list = np.zeros(mat.shape[0])
    y_label.setneg()  # 假设 setneg 方法用于设置负值
    select_list[random_list] = 1  # 标记随机选择的细胞

    # 添加用户选择的细胞
    select_list[selected] = 1
    selected_val = len(val) - 1  # 为用户选择的细胞分配一个类别
    print("selected_val: ", selected_val)

    # 更新标签列表
    mat_list = mat.tolist()
    for t in range(len(selected)):
        mat_list[selected[t]] = selected_val  # 将用户选择的细胞标记为选定的类别
    mat = pd.Categorical(mat_list)

    # 更新标签矩阵
    for i in range(mat.shape[0]):
        if select_list[i]:
            y_label.editval2(i, val[mat[i]])  # 假设 editval2 方法用于编辑标签值

    # 标签传播
    y_pred = LPA.mat(mat.shape[0], len(val))
    y_new = LPA.mat(mat.shape[0], len(val))
    LPA.labelPropagation(X, y_label, y_pred, y_new, 0.5, 1000)  # 假设 labelPropagation 是标签传播的实现

    # 处理预测结果
    y_res = np.zeros(mat.shape[0])
    if do_correct:
        # 使用 y_new 结果
        for i in range(mat.shape[0]):
            y_res[i] = y_new.getval(i, 0)  # 假设 getval 方法用于获取值
    else:
        # 使用 y_pred 结果
        for i in range(mat.shape[0]):
            y_res[i] = y_pred.getval(i, 0)

    # 过滤结果，返回所选类别的细胞索引
    y_res = pd.Series(y_res)
    y_res = y_res[y_res == selected_val]
    
    # 测试输出
    print("Selected cells:", selected)
    print("Unique labels in mat:", np.unique(mat))
    print("Value mapping:", val)
    print("Predicted indices for selected value:", list(y_res.index))
    
    return list(y_res.index)


import os
import logging
import scanpy as sc
import h5py
import scipy.sparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraphSAGE(torch.nn.Module):
    """GraphSAGE模型实现"""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


def load_data(adata_path, device):
    """从h5ad文件加载数据"""
    try:
        # 加载h5ad文件
        logger.info(f"Loading data from {adata_path}")
        adata = sc.read_h5ad(adata_path)

        # 加载特征矩阵
        features = torch.tensor(adata.X.todense(), dtype=torch.float)

        # 加载图结构
        mat = adata.obsp['connectivities']
        if not scipy.sparse.issparse(mat):
            mat = scipy.sparse.csr_matrix(mat)

        # 转换为COO格式并构建边索引
        coo = mat.tocoo()
        edge_index = torch.tensor([coo.row, coo.col], dtype=torch.long)

        # 处理标签: 将字符串标签转换为数值索引
        unique_labels = pd.Categorical(adata.obs['annotation'])
        labels = torch.tensor(unique_labels.codes, dtype=torch.long)

        # 将数据移至指定设备
        features = features.to(device)
        edge_index = edge_index.to(device)
        labels = labels.to(device)

        return features, edge_index, labels

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def GNNRefine(adata_path, selected, hidden_dim=64, n_epochs=100, train_ratio=0.1):
    """
    使用GraphSAGE进行细胞类型预测的函数

    参数:
    adata_path: h5ad文件路径
    selected: 用户选择的细胞索引列表
    hidden_dim: GNN隐藏层维度
    n_epochs: 训练轮数
    train_ratio: 随机选取的训练集比例

    返回:
    预测为目标类别的细胞索引列表
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 加载数据
    logger.info("Loading data...")
    features, edge_index, original_labels = load_data(adata_path, device)
    num_nodes = features.shape[0]

    # 获取原始标签的最大值，用户选择的细胞将被标记为 max_label + 1
    max_label = original_labels.max().item()
    target_label = max_label + 1
    num_classes = max_label + 2  # 包括原有标签和新的目标标签

    # 创建训练标签：复制原始标签，并为选中的细胞分配新标签
    y = original_labels.clone()
    y[selected] = target_label

    # 准备训练掩码：随机选择10%的细胞
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # 排除已选择的细胞，从剩余细胞中随机选择
    available_indices = list(set(range(num_nodes)) - set(selected))
    num_samples = int(num_nodes * train_ratio)
    random_indices = np.random.choice(available_indices, num_samples, replace=False)

    # 设置训练掩码
    train_mask[random_indices] = True  # 随机选取的训练样本
    train_mask[selected] = True  # 用户选择的细胞

    # 将训练掩码移至设备
    train_mask = train_mask.to(device)

    # 初始化模型
    model = GraphSAGE(
        in_channels=features.shape[1],
        hidden_channels=hidden_dim,
        out_channels=num_classes
    ).to(device)

    # 训练模型
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    logger.info("Starting training...")
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = model(features, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                pred = out[train_mask].max(1)[1]
                accuracy = pred.eq(y[train_mask]).sum().item() / train_mask.sum().item()
                logger.info(f'Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

    # 预测
    logger.info("Starting prediction...")
    model.eval()
    with torch.no_grad():
        out = model(features, edge_index)
        pred = out.max(1)[1].cpu().numpy()
        # 获取预测为目标标签的所有细胞索引
        predicted_indices = np.where(pred == target_label)[0]

        # 确保用户选择的细胞一定在结果中
        predicted_indices = np.unique(np.concatenate([predicted_indices, selected]))

    logger.info(f"Prediction completed. Found {len(predicted_indices)} similar cells")
    return list(predicted_indices)