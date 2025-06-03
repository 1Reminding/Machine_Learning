import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from lassoview import GNNRefine, LPARefine  # 确保GNNRefine代码保存在gnn_visualization.py中

# 读取 .h5ad 文件
adata_path = 'data/dataset1.h5ad'

# 读取 .txt 文件
with open('data/test/9.txt', 'r') as file:
    lines = file.readlines()

# 解析第一组数据
selected_cells = ast.literal_eval(lines[0].strip())
selected_cells = list(map(int, selected_cells))

# 可视化
def plot_cells(adata, selected_cells, predicted_indices, method_name="GNN"):
    if 'X_umap' in adata.obsm:
        coords = adata.obsm['X_umap']
    else:
        # 如果没有UMAP坐标，计算UMAP
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        coords = adata.obsm['X_umap']

    df = pd.DataFrame(coords, columns=['UMAP1', 'UMAP2'])

    selected_mask = np.zeros(len(df), dtype=bool)
    predicted_mask = np.zeros(len(df), dtype=bool)

    selected_mask[selected_cells] = True
    predicted_mask[predicted_indices] = True

    plt.figure(figsize=(12, 6))

    # 选定细胞图
    plt.subplot(1, 2, 1)
    plt.scatter(df['UMAP1'], df['UMAP2'], color='gray', alpha=0.3, edgecolors='w', s=20, label='Other Cells')
    plt.scatter(df['UMAP1'][selected_mask], df['UMAP2'][selected_mask], color='red', alpha=0.6, edgecolors='w', s=50,
                label='Selected Cells')
    plt.title('Selected Cells')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(loc='best')
    plt.grid(True)

    # 预测细胞图
    plt.subplot(1, 2, 2)
    plt.scatter(df['UMAP1'], df['UMAP2'], color='gray', alpha=0.3, edgecolors='w', s=20, label='Other Cells')
    plt.scatter(df['UMAP1'][predicted_mask], df['UMAP2'][predicted_mask], color='blue', alpha=0.6, edgecolors='w', s=50,
                label=f'{method_name} Predicted Cells')
    plt.title(f'{method_name} Predicted Cells')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(loc='best')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{method_name.lower()}_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.close()


# 比较GNN和LPA的结果
adata = sc.read(adata_path)

# 使用 GNNRefine 进行预测
predicted_indices = GNNRefine(adata_path, selected_cells)
plot_cells(adata, selected_cells, predicted_indices, method_name="GNN")

# LPA预测（可选）
lpa_predicted_indices = LPARefine(adata_path, selected_cells, function="h5adfile", do_correct=False)
plot_cells(adata, selected_cells, lpa_predicted_indices, method_name="LPA")