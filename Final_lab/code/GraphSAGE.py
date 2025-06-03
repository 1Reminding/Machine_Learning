import os
import logging
import time
import random
import scanpy as sc
import h5py
import scipy.sparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import adjusted_rand_score

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置参数
CONFIG = {
    'hidden_channels': 128,
    'learning_rate': 0.01,
    'weight_decay': 5e-4,
    'epochs': 55,
    'mask_ratio': 0.9,
    'runs_per_noise': 100,  # 每个错误率运行的次数
    'random_seed': 42
}

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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

def load_data(feature_path, graph_path, device):
    """加载数据集"""
    try:
        # 加载特征矩阵
        logger.info(f"Loading features from {feature_path}")
        data = sc.read_h5ad(feature_path)
        features = torch.tensor(data.X.todense(), dtype=torch.float)
        features = features.to(device)

        # 加载图结构
        logger.info(f"Loading graph structure from {graph_path}")
        with h5py.File(graph_path, 'r') as f:
            # 加载邻接矩阵
            group = f['obsp']['connectivities']
            adj_data = group['data'][:]
            indices = group['indices'][:]
            indptr = group['indptr'][:]
            shape = (
                f['obsp']['connectivities'].attrs['shape'][0],
                f['obsp']['connectivities'].attrs['shape'][1]
            )

            # 构建稀疏矩阵
            mat = scipy.sparse.csr_matrix((adj_data, indices, indptr), shape=shape)
            coo = mat.tocoo()

            # 构建边索引
            edge_index = torch.tensor([coo.row, coo.col], dtype=torch.long)
            edge_index = edge_index.to(device)

            # 加载标签
            obs_group = f['obs']
            if "codes" in obs_group['annotation']:
                labels = obs_group['annotation']['codes'][:]
            else:
                labels = obs_group['annotation'][:]

            labels = torch.tensor(labels, dtype=torch.long)
            labels = labels.to(device)

        return features, edge_index, labels

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def prepare_masks(labels, mask_ratio, noise_ratio, device):
    """准备训练和测试掩码"""
    num_nodes = labels.size(0)
    num_masked = int(num_nodes * mask_ratio)

    # 创建掩码
    masked_indices = np.random.choice(num_nodes, num_masked, replace=False)
    masked_labels = labels.clone()
    masked_labels[masked_indices] = -1

    # 添加噪声标签
    remaining_indices = np.setdiff1d(np.arange(num_nodes), masked_indices)
    error_indices = np.random.choice(
        remaining_indices,
        int(len(remaining_indices) * noise_ratio),
        replace=False
    )

    # 为选定的样本分配错误标签
    for idx in error_indices:
        original_label = labels[idx].item()
        wrong_label = np.random.choice(
            np.setdiff1d(np.arange(labels.max().item() + 1), [original_label])
        )
        masked_labels[idx] = wrong_label

    # 创建训练和测试掩码
    train_mask = (masked_labels != -1)
    test_mask = ~train_mask

    return masked_labels.to(device), train_mask.to(device), test_mask.to(device)

def rectify_predictions(pred, edge_index, labels, train_mask):
    """使用图结构修正预测结果"""
    rectified_pred = pred.clone()
    
    # 构建邻接表
    adj_dict = {}
    for i in range(edge_index.size(1)):
        src, dst = edge_index[:, i]
        if src.item() not in adj_dict:
            adj_dict[src.item()] = []
        adj_dict[src.item()].append(dst.item())

    # 对训练集中的节点进行修正
    for node in range(len(pred)):
        if train_mask[node]:
            if node in adj_dict:
                # 获取邻居的预测标签
                neighbor_preds = [pred[neighbor].item() for neighbor in adj_dict[node]]
                if neighbor_preds:
                    # 使用最常见的邻居标签
                    most_common = max(set(neighbor_preds), key=neighbor_preds.count)
                    rectified_pred[node] = most_common

    return rectified_pred

def train_and_evaluate(features, edge_index, labels, noise_rate, device):
    """训练模型并评估结果"""
    # 准备掩码和添加噪声
    masked_labels, train_mask, test_mask = prepare_masks(
        labels, 
        CONFIG['mask_ratio'],
        noise_rate,
        device
    )
    
    # 初始化模型
    model = GraphSAGE(
        features.size(1),
        CONFIG['hidden_channels'],
        labels.max().item() + 1
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # 训练模型
    model.train()
    for epoch in range(CONFIG['epochs']):
        optimizer.zero_grad()
        out = model(features, edge_index)
        loss = F.cross_entropy(out[train_mask], masked_labels[train_mask])
        loss.backward()
        optimizer.step()

    # 评估
    model.eval()
    with torch.no_grad():
        pred = model(features, edge_index).argmax(dim=1)
        
    # 计算原始ARI
    ari_o = adjusted_rand_score(labels.cpu().numpy(), pred.cpu().numpy())
    
    # 应用修正并计算修正后的ARI
    rectified_pred = rectify_predictions(pred, edge_index, labels, train_mask)
    ari_r = adjusted_rand_score(labels.cpu().numpy(), rectified_pred.cpu().numpy())
    
    return ari_o, ari_r

def main():
    """主函数"""
    # 设置随机种子
    set_seed(CONFIG['random_seed'])

    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    try:
        # 加载数据
        features, edge_index, labels = load_data(
            'data/dataset.h5ad',
            'data/dataset1.h5ad',
            device
        )

        # 创建DataFrame来存储结果
        results = []
        
        # 错误率从0到0.95，步长为0.05
        for noise_rate in np.arange(0, 1.0, 0.05):
            logger.info(f"Processing noise rate: {noise_rate:.2f}")

            # 对每个错误率运行多次
            for run in range(CONFIG['runs_per_noise']):
                start_time = time.perf_counter()

                # 训练和评估
                ari_o, ari_r = train_and_evaluate(
                    features,
                    edge_index,
                    labels,
                    noise_rate,
                    device
                )

                end_time = time.perf_counter()
                execution_time = end_time - start_time

                # 记录结果
                results.append({
                    'MR': f"{noise_rate:.2f}",  # 错误率
                    'ARI_o': ari_o,             # 原始ARI
                    'ARI_r': ari_r,             # 修正后ARI
                    'Time': execution_time       # 执行时间
                })

                logger.info(f"Run {run + 1}/{CONFIG['runs_per_noise']} completed. "
                          f"ARI_o: {ari_o:.4f}, ARI_r: {ari_r:.4f}, "
                          f"Time: {execution_time:.4f}s")

        # 将结果转换为DataFrame并保存
        df = pd.DataFrame(results)
        df.to_csv("mistakegnn.tsv", sep='\t', index=True)
        logger.info("Results saved to mistake.tsv")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()