import numpy as np
import random
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 
# 数据读取
path = "winequality-white.csv"
def read_data(path):
    # 从CSV文件中读取数据
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    label = data[:, -1]  # 标签是数据集的最后一列
    data = data[:, :-1]  # 特征是除最后一列外的其他列
    return np.array(data), np.array(label)
# 数据标准化
def standardize(data):
    # 标准化处理数据：均值为0，标准差为1
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


# 数据划分
def split_data(data, label, test_rate):
    train_index = []
    test_index = []
    label_ids = {}

    # 统计每个标签对应的样本索引
    for i in range(len(label)):
        label_val = label[i]
        if label_val not in label_ids:
            label_ids[label_val] = []
        label_ids[label_val].append(i)

    # 随机抽取测试样本
    for key, value in label_ids.items():
        sample_num = int(test_rate * len(value))
        sample_list = random.sample(value, sample_num)
        test_index.extend(sample_list)

    train_index = list(set(range(len(label))) - set(test_index))

    # 打乱索引以保证随机性
    random.shuffle(train_index)
    random.shuffle(test_index)

    train_data = [data[i] for i in train_index]
    train_label = [label[i] for i in train_index]
    test_data = [data[i] for i in test_index]
    test_label = [label[i] for i in test_index]

    return np.array(train_data), np.array(train_label), np.array(test_data), np.array(test_label)
# 数据处理
def data_process(path):
    data, label = read_data(path)  # 读取数据
    data = standardize(data)  # 标准化数据
    # 加入偏置项，所有数据的第一列都为1，用于线性回归的偏置项
    data = np.hstack((np.ones((data.shape[0], 1)), data))
    train_data, train_label, test_data, test_label = split_data(data, label, 0.2)
    return train_data, train_label, test_data, test_label
# 计算MSE
def MSE(weight, x, y):
    difference = y - np.dot(x, weight)
    error = np.array([num * num for num in difference])
    mse = error.mean()
    return mse
# 计算随机梯度
def Stochastic_gradient(weight, x, y):
    difference = y - np.dot(x, weight)
    delta = np.dot(np.transpose(difference), x)
    return delta


# 随机梯度下降
def Stochastic_gradient_descent(data, label, learning_rate, epochs=100, epsilon=1e-7):
    weight = np.random.rand(data.shape[1])
    mses = [MSE(weight, data, label)]

    for epoch in range(epochs):
        for i in range(data.shape[0]):
            x = data[i, :]
            y = label[i]
            delta = Stochastic_gradient(weight, x, y)
            weight = weight + learning_rate * delta
            mse = MSE(weight, data, label)
            mses.append(mse)
            print(f"Epoch {epoch + 1}/{epochs}, SGD Train MSE: {mse:.4f}")
            if abs(mses[-2] - mses[-1]) < epsilon:
                return mses, weight
    print(f"最终收敛的 SGD Train MSE: {mse:.4f}, 学习率: {learning_rate}")
    return mses, weight

# 计算批量梯度
def batch_gradient(weight, x, y):
    difference = y - np.dot(x, weight)
    delta = np.dot(np.transpose(difference), x) / x.shape[0]  # 求平均梯度
    return delta

# 批量梯度下降
def batch_gradient_descent(data, label, learning_rate, epochs=3000, epsilon=1e-7):
    weight = np.random.rand(data.shape[1])
    mses = [MSE(weight, data, label)]

    for epoch in range(epochs):
        delta = batch_gradient(weight, data, label)
        weight = weight + learning_rate * delta
        mse = MSE(weight, data, label)
        mses.append(mse)
        print(f"Epoch {epoch + 1}/{epochs}, BGD Train MSE: {mse:.4f}")
        if abs(mses[-2] - mses[-1]) < epsilon:
            break
    print(f"最终收敛的 BGD Train MSE: {mse:.4f}, 学习率: {learning_rate}")
    return mses, weight
# 批量梯度下降的线性回归
def regression_BGD(path, learning_rate):
    train_data, train_label, test_data, test_label = data_process(path)
    mses, weight = batch_gradient_descent(train_data, train_label, learning_rate)
    mse_test = MSE(weight, test_data, test_label)
    mse_train = MSE(weight, train_data, train_label)
    print(f"测试集上的最终 BGD MSE: {mse_test:.4f}")
    return mse_test, mse_train, mses
# 随机梯度下降的线性回归
def regression_SGD(path, learning_rate):
    train_data, train_label, test_data, test_label = data_process(path)
    mses, weight = Stochastic_gradient_descent(train_data, train_label, learning_rate)
    mse_test = MSE(weight, test_data, test_label)
    mse_train = MSE(weight, train_data, train_label)
    print(f"测试集上的最终 SGD MSE: {mse_test:.4f}")
    return mse_test, mse_train, mses

# 展示批量梯度下降的训练结果
def show_BGD(path, learning_rate):
    mse_test_BGD, mse_train_BGD, mses_BGD = regression_BGD(path, learning_rate)
    print('学习率 :', learning_rate)
    print('测试集上均方误差 :', mse_test_BGD)
    print('训练集上均方误差:', mse_train_BGD)

    plt.figure()
    k = [x for x in range(len(mses_BGD))]
    plt.plot(k, mses_BGD, color='red')
    plt.xlabel('epoch')
    plt.ylabel('均方误差(MSE)')
    plt.title("批量梯度下降(BGD)epoch-mse")
    plt.show()

# 展示随机梯度下降的训练结果
def show_SGD(path, learning_rate):
    mse_test_SGD, mse_train_SGD, mses_SGD = regression_SGD(path, learning_rate)
    print('学习率 :', learning_rate)
    print('测试集上均方误差 :', mse_test_SGD)
    print('训练集上均方误差:', mse_train_SGD)

    plt.figure()
    k = [x for x in range(len(mses_SGD))]
    plt.plot(k, mses_SGD, color='blue')
    plt.xlabel('epoch')
    plt.ylabel('均方误差(MSE)')
    plt.title("随机梯度下降(SGD)epoch-mse")
    plt.show()

# 运行代码并展示结果
show_BGD(path, 0.0005)
show_SGD(path, 0.0005)


# 用于收集不同学习率的 MSE 结果，并找到最佳学习率
def collect_mse_results(path, learning_rates):
    bgd_train_mses = []
    bgd_test_mses = []
    sgd_train_mses = []
    sgd_test_mses = []

    for lr in learning_rates:
        mse_test_BGD, mse_train_BGD, _ = regression_BGD(path, lr)
        mse_test_SGD, mse_train_SGD, _ = regression_SGD(path, lr)

        bgd_train_mses.append(mse_train_BGD)
        bgd_test_mses.append(mse_test_BGD)
        sgd_train_mses.append(mse_train_SGD)
        sgd_test_mses.append(mse_test_SGD)

    # 找到测试集上 MSE 最小的学习率
    best_bgd_lr = learning_rates[np.argmin(bgd_test_mses)]
    best_sgd_lr = learning_rates[np.argmin(sgd_test_mses)]

    return bgd_train_mses, bgd_test_mses, sgd_train_mses, sgd_test_mses, best_bgd_lr, best_sgd_lr

# 绘制所有学习率的 MSE 结果
def plot_mse_results(learning_rates, bgd_train_mses, bgd_test_mses, sgd_train_mses, sgd_test_mses, best_bgd_lr, best_sgd_lr):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 批量梯度下降图像
    ax1.plot(learning_rates, bgd_train_mses, label="Batch GD - Train MSE", color="blue", marker="o")
    ax1.plot(learning_rates, bgd_test_mses, label="Batch GD - Test MSE", color="orange", marker="o", linestyle="--")
    ax1.set_xscale('log')
    ax1.set_xlabel("Learning Rate")
    ax1.set_ylabel("Mean Squared Error (MSE)")
    ax1.set_title(f"MSE for Batch Gradient Descent (Best LR: {best_bgd_lr})")
    ax1.legend()

    # 随机梯度下降图像
    ax2.plot(learning_rates, sgd_train_mses, label="SGD - Train MSE", color="green", marker="s")
    ax2.plot(learning_rates, sgd_test_mses, label="SGD - Test MSE", color="red", marker="s", linestyle="--")
    ax2.set_xscale('log')
    ax2.set_xlabel("Learning Rate")
    ax2.set_ylabel("Mean Squared Error (MSE)")
    ax2.set_title(f"MSE for Stochastic Gradient Descent (Best LR: {best_sgd_lr})")
    ax2.legend()

    plt.tight_layout()
    plt.show()

# 定义不同的学习率进行测试
learning_rates = [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.007, 0.01]

# 收集 MSE 结果
bgd_train_mses, bgd_test_mses, sgd_train_mses, sgd_test_mses, best_bgd_lr, best_sgd_lr = collect_mse_results(path, learning_rates)

# 绘制学习率对比图
plot_mse_results(learning_rates, bgd_train_mses, bgd_test_mses, sgd_train_mses, sgd_test_mses, best_bgd_lr, best_sgd_lr)

# 输出最佳学习率
print(f"最佳学习率 (BGD): {best_bgd_lr}")
print(f"最佳学习率 (SGD): {best_sgd_lr}")