import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

path = "winequality-white.csv"


# 数据标准化
def standardize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


# 数据集读取
def read_data(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    label = data[:, -1]  # 标签是最后一列
    data = data[:, :-1]  # 特征是其他列
    return np.array(data), np.array(label)


# 岭回归解析解
def ridge_regression(X_train, y_train, lambd):
    XTX = np.dot(X_train.T, X_train)
    lambda_identity = lambd * np.eye(X_train.shape[1])
    inverse_term = np.linalg.inv(XTX + lambda_identity)
    w = np.dot(inverse_term, np.dot(X_train.T, y_train))
    return w


# 计算MSE
def MSE(weight, X, y):
    predictions = np.dot(X, weight)
    error = y - predictions
    mse = np.mean(error ** 2)
    return mse


# 运行岭回归并计算误差，记录权重
def run_ridge_regression(X_train, X_test, y_train, y_test, lambd):
    # 计算岭回归权重
    w_ridge = ridge_regression(X_train, y_train, lambd)

    # 计算训练集和测试集的MSE
    train_mse = MSE(w_ridge, X_train, y_train)
    test_mse = MSE(w_ridge, X_test, y_test)

    # 输出权重信息
    print(f"正则化参数 λ = {lambd}")
    print(f"岭回归权重: {w_ridge}")
    print(f"训练集平均误差 (MSE)：{train_mse:.4f}")
    print(f"测试集平均误差 (MSE)：{test_mse:.4f}")
    return train_mse, test_mse, w_ridge


# 交叉验证选择最佳 λ，并绘制权重曲线
def cross_validate_ridge_regression(path, lambdas, folds=5):
    # 读取数据并标准化
    data, label = read_data(path)
    data = standardize(data)
    data = np.hstack((np.ones((data.shape[0], 1)), data))  # 加入偏置项

    # 使用K折交叉验证
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    avg_train_errors = []
    avg_test_errors = []
    best_lambda = None
    lowest_test_mse = float('inf')

    all_weights = []

    for lambd in lambdas:
        train_errors = []
        test_errors = []
        weights_per_lambda = []

        for train_index, test_index in kf.split(data):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]

            # 计算当前 λ 下的MSE和权重
            train_mse, test_mse, weights = run_ridge_regression(X_train, X_test, y_train, y_test, lambd)
            train_errors.append(train_mse)
            test_errors.append(test_mse)
            weights_per_lambda.append(weights)

        avg_train_mse = np.mean(train_errors)
        avg_test_mse = np.mean(test_errors)
        avg_train_errors.append(avg_train_mse)
        avg_test_errors.append(avg_test_mse)

        # 存储该λ下所有折的权重平均值
        avg_weights = np.mean(weights_per_lambda, axis=0)
        all_weights.append(avg_weights)

        if avg_test_mse < lowest_test_mse:
            lowest_test_mse = avg_test_mse
            best_lambda = lambd

        print(f"λ = {lambd}, 平均训练误差: {avg_train_mse:.4f}, 平均测试误差: {avg_test_mse:.4f}")

    # 绘制误差曲线
    plt.figure(figsize=(10, 5))
    plt.plot(lambdas, avg_train_errors, label="Train MSE", color="blue", marker="o")
    plt.plot(lambdas, avg_test_errors, label="Test MSE", color="red", marker="s", linestyle="--")
    plt.xscale('log')
    plt.xlabel("Regularization Parameter (λ)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Cross-Validation: MSE vs. λ for Ridge Regression")
    plt.legend()
    plt.show()

    # 绘制权重更新曲线，并根据不同正则化参数注释
    plot_weight_update(lambdas, all_weights)

    print(f"\n最佳的正则化参数 λ = {best_lambda}, 对应的测试集 MSE = {lowest_test_mse:.4f}")

    return best_lambda


def plot_weight_update(lambdas, all_weights):
    plt.figure(figsize=(10, 6))
    all_weights = np.array(all_weights)

    # 绘制每一个特征的权重随不同λ的变化曲线
    for i in range(all_weights.shape[1]):
        plt.plot(lambdas, all_weights[:, i], label=f"特征权重 {i}")

    plt.xscale('log')
    plt.xlabel("Regularization Parameter (λ)")
    plt.ylabel("Weights")
    plt.title("Weights vs. λ in Ridge Regression")

    # 设置图例为不同的正则化参数 (λ)
    plt.legend(loc='best', title="正则化参数 (λ)")
    plt.grid(True)
    plt.show()


# 运行交叉验证，并选择最佳 λ
lambdas = [0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 15, 20, 50]  # 这里是设置的不同正则化参数
best_lambda = cross_validate_ridge_regression(path, lambdas)