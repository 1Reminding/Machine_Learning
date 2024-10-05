# import numpy as np
# from sklearn.metrics import accuracy_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 加载数据
# file_path = 'semeion.data'
# data = np.loadtxt(file_path)
#
# # 提取特征和标签
# X = data[:, :256]  # 前256列是特征
# y = np.argmax(data[:, 256:], axis=1)  # 后面九个数字表示类别
#
#
# # 手动实现的kNN算法
# def manual_knn(X_train, X_test, y_train, k):
#     # 计算欧几里得距离
#     distances = np.sqrt(np.sum((X_train - X_test) ** 2, axis=1))
#     nearest_neighbors = np.argsort(distances)[:k]
#     nearest_labels = y_train[nearest_neighbors]
#     return np.bincount(nearest_labels).argmax()
#
#
# # 手动实现留一法并计算精度
# def manual_leave_one_out_knn(k_values):
#     accuracies = []
#
#     for k in k_values:
#         y_true = []
#         y_pred = []
#
#         print(f"\nEvaluating k={k} using manual Leave-One-Out...")
#         for i in range(len(X)):
#             if (i + 1) % 1000 == 0 or i == len(X) - 1:
#                 print(f"Processing sample {i + 1}/{len(X)}")
#
#             # 将第i个样本作为测试集，剩余的样本作为训练集
#             X_test = X[i]  # 第i行作为测试数据
#             y_test = y[i]
#             X_train = np.delete(X, i, axis=0)  # 删除第i行数据
#             y_train = np.delete(y, i, axis=0)
#
#             # 使用手动实现的kNN进行分类
#             pred = manual_knn(X_train, X_test, y_train, k)
#             y_pred.append(pred)
#             y_true.append(y_test)
#
#         # 计算精度
#         accuracy = accuracy_score(y_true, y_pred)
#         print(f"Accuracy for k={k}: {accuracy:.4f}")
#         accuracies.append((k, accuracy))
#
#     return accuracies, y_true, y_pred
#
#
# # 设置k值
# k_values = [5, 9, 13]
#
# # 运行手动实现的kNN并获取精度和预测结果
# accuracies, all_y_true, all_y_pred = manual_leave_one_out_knn(k_values)
#
# # 汇总输出
# print("\nSummary of accuracies:")
# for k, acc in accuracies:
#     print(f"k={k}, Accuracy={acc:.4f}")
#
#
# # 可视化部分
# def visualize_results(accuracies, y_true, y_pred, best_k):
#     # 1. 绘制k值与准确率的关系图
#     ks, accs = zip(*accuracies)
#     plt.figure(figsize=(8, 6))
#     plt.plot(ks, accs, marker='o')
#     plt.title('k值与准确率的关系')
#     plt.xlabel('k值')
#     plt.ylabel('准确率')
#     plt.xticks(ks)
#     plt.grid(True)
#     plt.show()
#
#     # 2. 绘制混淆矩阵
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=np.unique(y), yticklabels=np.unique(y))
#     plt.title(f'混淆矩阵 (k={best_k})')
#     plt.xlabel('预测类别')
#     plt.ylabel('真实类别')
#     plt.show()
#
#
# # 选择最佳k值（假设第一个k为最佳，或根据实际情况选择）
# # 这里以最高准确率对应的k为最佳k
# best_k, best_acc = max(accuracies, key=lambda x: x[1])
# print(f"\n最佳k值为 k={best_k}，对应的准确率为 {best_acc:.4f}")
#
#
# # 重新运行一次留一法以获取最佳k的混淆矩阵
# def get_predictions_for_k(k):
#     y_true = []
#     y_pred = []
#
#     print(f"\nCollecting predictions for k={k}...")
#     for i in range(len(X)):
#         if (i + 1) % 1000 == 0 or i == len(X) - 1:
#             print(f"Processing sample {i + 1}/{len(X)}")
#
#         X_test = X[i]
#         y_test = y[i]
#         X_train = np.delete(X, i, axis=0)
#         y_train = np.delete(y, i, axis=0)
#
#         pred = manual_knn(X_train, X_test, y_train, k)
#         y_pred.append(pred)
#         y_true.append(y_test)
#
#     return y_true, y_pred
#
#
# # 获取最佳k的预测结果
# best_y_true, best_y_pred = get_predictions_for_k(best_k)
#
# # 可视化结果
# visualize_results(accuracies, best_y_true, best_y_pred, best_k)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.font_manager as fm

# 设置字体为SimHei（黑体），以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# 加载数据
file_path = 'semeion.data'
data = np.loadtxt(file_path)

# 提取特征和标签
X = data[:, :256]  # 前256列是特征
y = np.argmax(data[:, 256:], axis=1)  # 后面九个数字表示类别

# 手动实现的kNN算法
def manual_knn(X_train, X_test, y_train, k):
    # 计算欧几里得距离
    distances = np.sqrt(np.sum((X_train - X_test) ** 2, axis=1))
    nearest_neighbors = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest_neighbors]
    return np.bincount(nearest_labels).argmax()

# 手动实现留一法并计算精度
def manual_leave_one_out_knn(k_values):
    accuracies = []
    all_y_true = []
    all_y_pred = []

    for k in k_values:
        y_true = []
        y_pred = []

        print(f"\nEvaluating k={k} using manual Leave-One-Out...")
        for i in range(len(X)):
            # 将第i个样本作为测试集，剩余的样本作为训练集
            X_train = np.delete(X, i, axis=0)  # 删除第i行数据
            y_train = np.delete(y, i, axis=0)
            X_test = X[i]  # 第i行作为测试数据
            y_test = y[i]

            # 使用手动实现的kNN进行分类
            pred = manual_knn(X_train, X_test, y_train, k)
            y_pred.append(pred)
            y_true.append(y_test)

        # 计算精度
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy for k={k}: {accuracy:.4f}")
        accuracies.append((k, accuracy))

        if k == 9:  # 选定 k=9 时绘制混淆矩阵
            all_y_true = y_true
            all_y_pred = y_pred

    return accuracies, all_y_true, all_y_pred

# 结果可视化函数
def plot_accuracies(k_values, accuracies):
    # 提取 k 值和对应的准确度
    ks = [k for k, acc in accuracies]
    accs = [acc for k, acc in accuracies]

    # 绘制图表
    plt.figure(figsize=(8, 6))
    plt.plot(ks, accs, marker='o', linestyle='-', color='b')
    plt.title('Accuracy vs k in kNN with Leave-One-Out Cross-Validation')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

# 混淆矩阵可视化
def plot_confusion_matrix(y_true, y_pred, k):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'混淆矩阵 (k={k})')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()

# 设置k值
k_values = [5, 9, 13]

# 运行手动实现的kNN并获取精度
accuracies, all_y_true, all_y_pred = manual_leave_one_out_knn(k_values)

# 绘制准确度图
plot_accuracies(k_values, accuracies)

# 绘制混淆矩阵（选定 k=9）
plot_confusion_matrix(all_y_true, all_y_pred, k=9)
