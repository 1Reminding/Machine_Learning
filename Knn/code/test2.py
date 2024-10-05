import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm
import math

# 加载数据
file_path = 'semeion.data'
data = np.loadtxt(file_path)

# 提取特征和标签
X = data[:, :256]  # 前256列是特征
y = np.argmax(data[:, 256:], axis=1)  # 后面九个数字表示类别

# 定义留一法交叉验证
loo = LeaveOneOut()


# 手动实现的kNN算法
def manual_knn(X_train, X_test, y_train, k):
    distances = np.sqrt(np.sum((X_train - X_test) ** 2, axis=1))
    nearest_neighbors = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest_neighbors]
    # 返回出现最多的标签
    return np.bincount(nearest_labels).argmax()


# 计算概率混淆矩阵
def compute_probabilistic_confusion_matrix(conf_matrix):
    prob_conf_matrix = conf_matrix.astype(float)
    for i in range(conf_matrix.shape[0]):
        prob_conf_matrix[i] = conf_matrix[i] / np.sum(conf_matrix[i])
    return prob_conf_matrix


# 计算P_ij值
def compute_P_ij(p_ij, prob_matrix, m):
    return p_ij / np.sum(prob_matrix, axis=1)


# 计算详细的 CEN
def calculate_cen(conf_matrix):
    m = conf_matrix.shape[0]  # 类别数量
    prob_matrix = compute_probabilistic_confusion_matrix(conf_matrix)

    # 初始化每个类别的CEN值
    CENj = np.zeros(m)

    # 计算每个类别j的CEN_j
    for j in range(m):
        for k in range(m):
            if j != k:
                # 计算单个概率值
                P_jjk = prob_matrix[j, k] / np.sum(prob_matrix[j])
                P_kj = prob_matrix[k, j] / np.sum(prob_matrix[k])

                # 确保计算出的值是标量
                if np.isscalar(P_jjk) and np.isscalar(P_kj):
                    CENj[j] -= (P_jjk * np.log2((m - 1) * P_jjk + 1e-9) + P_kj * np.log2((m - 1) * P_kj + 1e-9))

    # 计算总体CEN
    total_Pj = np.sum(prob_matrix, axis=0) / (2 * np.sum(prob_matrix))
    overall_CEN = np.sum(total_Pj * CENj)

    return overall_CEN


# 对比手动实现kNN和sklearn的kNN
def compare_knn(k_values):
    results = []
    for k in k_values:
        y_true = []
        y_pred_manual = []
        y_pred_sklearn = []

        # sklearn的kNN分类器
        knn = KNeighborsClassifier(n_neighbors=k)

        print(f"Evaluating k={k}...")
        for train_index, test_index in tqdm(loo.split(X), total=len(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 手动实现kNN
            pred_manual = manual_knn(X_train, X_test[0], y_train, k)
            y_pred_manual.append(pred_manual)

            # sklearn实现kNN
            knn.fit(X_train, y_train)
            pred_sklearn = knn.predict(X_test)[0]
            y_pred_sklearn.append(pred_sklearn)

            y_true.append(y_test[0])

        # 计算性能指标
        acc_manual = accuracy_score(y_true, y_pred_manual)
        acc_sklearn = accuracy_score(y_true, y_pred_sklearn)
        nmi_manual = normalized_mutual_info_score(y_true, y_pred_manual)
        nmi_sklearn = normalized_mutual_info_score(y_true, y_pred_sklearn)

        # 计算混淆矩阵
        cm_manual = confusion_matrix(y_true, y_pred_manual)
        cm_sklearn = confusion_matrix(y_true, y_pred_sklearn)

        # 计算CEN
        cen_manual = calculate_cen(cm_manual)
        cen_sklearn = calculate_cen(cm_sklearn)

        results.append({
            "k": k,
            "ACC (Manual)": acc_manual,
            "ACC (sklearn)": acc_sklearn,
            "NMI (Manual)": nmi_manual,
            "NMI (sklearn)": nmi_sklearn,
            "CEN (Manual)": cen_manual,
            "CEN (sklearn)": cen_sklearn
        })
        # 在终端打印每个 k 值的结果
        print(f"Results for k={k}:")
        print(f"  手动实现 kNN - ACC: {acc_manual:.4f}, NMI: {nmi_manual:.4f}, CEN: {cen_manual:.4f}")
        print(f"  sklearn实现 kNN - ACC: {acc_sklearn:.4f}, NMI: {nmi_sklearn:.4f}, CEN: {cen_sklearn:.4f}")

    return results


# 设置k值
k_values = [5, 9, 13]

# 运行实验并比较结果
knn_comparison_results = compare_knn(k_values)

import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 可视化对比结果
def visualize_comparison_results(results):
    ks = [r["k"] for r in results]

    acc_manual = [r["ACC (Manual)"] for r in results]
    acc_sklearn = [r["ACC (sklearn)"] for r in results]
    nmi_manual = [r["NMI (Manual)"] for r in results]
    nmi_sklearn = [r["NMI (sklearn)"] for r in results]
    cen_manual = [r["CEN (Manual)"] for r in results]
    cen_sklearn = [r["CEN (sklearn)"] for r in results]

    # ACC对比
    plt.figure(figsize=(10, 6))
    plt.plot(ks, acc_manual, label='手动实现的kNN - ACC', marker='o')
    plt.plot(ks, acc_sklearn, label='sklearn实现的kNN - ACC', marker='o')
    plt.xlabel('k值', fontsize=12)
    plt.ylabel('ACC (精度)', fontsize=12)
    plt.title('手动实现与sklearn kNN的精度对比', fontsize=15)
    plt.legend()
    plt.show()

    # NMI对比
    plt.figure(figsize=(10, 6))
    plt.plot(ks, nmi_manual, label='手动实现的kNN - NMI', marker='o')
    plt.plot(ks, nmi_sklearn, label='sklearn实现的kNN - NMI', marker='o')
    plt.xlabel('k值', fontsize=12)
    plt.ylabel('NMI (归一化互信息)', fontsize=12)
    plt.title('手动实现与sklearn kNN的NMI对比', fontsize=15)
    plt.legend()
    plt.show()

    # CEN对比
    plt.figure(figsize=(10, 6))
    plt.plot(ks, cen_manual, label='手动实现的kNN - CEN', marker='o')
    plt.plot(ks, cen_sklearn, label='sklearn实现的kNN - CEN', marker='o')
    plt.xlabel('k值', fontsize=12)
    plt.ylabel('CEN (混淆熵)', fontsize=12)
    plt.title('手动实现与sklearn kNN的CEN对比', fontsize=15)
    plt.legend()
    plt.show()


# 可视化实验结果
visualize_comparison_results(knn_comparison_results)
