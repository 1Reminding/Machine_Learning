import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
def load_data(file_path):
    data = np.loadtxt(file_path)
    X = data[:, :256]  # 图像数据
    y = data[:, 256:]  # 标签
    y = np.argmax(y, axis=1)  # 将one-hot编码的标签转换为整数标签
    X = X.reshape(-1, 16, 16, 1)  # 将数据重塑为16x16的单通道图像
    return X, y

# 图像旋转
def rotate_images(images, angle):
    rotated_images = []
    for img in images:
        M = cv2.getRotationMatrix2D((8, 8), angle, 1.0)  # 以(8,8)为中心旋转
        rotated_img = cv2.warpAffine(img.squeeze(), M, (16, 16))  # 旋转图像
        rotated_img = rotated_img.reshape(16, 16, 1)  # 重新调整形状
        rotated_images.append(rotated_img)
    return np.array(rotated_images)

# 构建CNN模型
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(16, 16, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')  # 输出10类，分别对应0-9
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 加载原始数据（不进行旋转增强）
X_original, y_original = load_data('semeion.data')

# 划分训练集和测试集（原始数据集）
X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(
    X_original, y_original, test_size=0.2, random_state=42
)

# 调整数据格式以适应CNN输入（原始数据集）
X_train_original = X_train_original.astype('float32') / 255.0  # 归一化处理
X_test_original = X_test_original.astype('float32') / 255.0

# 创建并训练CNN模型（原始数据集）
cnn_model_original = create_cnn_model()
history_original = cnn_model_original.fit(X_train_original, y_train_original, epochs=55, batch_size=32,
                                          validation_data=(X_test_original, y_test_original))

# 模型评估（原始数据集）
loss_original, accuracy_original = cnn_model_original.evaluate(X_test_original, y_test_original)
print(f'CNN模型在原始数据集上的准确率: {accuracy_original:.4f}, loss: {loss_original:.4f}')

# 加载数据并进行旋转增强
X, y = load_data('semeion.data')

# 数据增强 - 旋转图像
X_rotated_15 = rotate_images(X, 15)  # 左上方向旋转15度
X_rotated_345 = rotate_images(X, -15)  # 右下方向旋转15度

# 合并原始数据和旋转后的数据
X_augmented = np.concatenate([X, X_rotated_15, X_rotated_345])
y_augmented = np.concatenate([y, y, y])

# 划分训练集和测试集（旋转增强数据集）
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42)

# 调整数据格式以适应CNN输入（旋转增强数据集）
X_train = X_train.astype('float32') / 255.0  # 归一化处理
X_test = X_test.astype('float32') / 255.0

# 创建并训练CNN模型（旋转增强数据集）
cnn_model_augmented = create_cnn_model()
history_augmented = cnn_model_augmented.fit(X_train, y_train, epochs=55, batch_size=32,
                                            validation_data=(X_test, y_test))

# 模型评估（旋转增强数据集）
loss_augmented, accuracy_augmented = cnn_model_augmented.evaluate(X_test, y_test)
print(f'CNN模型在旋转增强数据集上的准确率: {accuracy_augmented:.4f}, loss: {loss_augmented:.4f}')

# 绘制两个数据集的准确率和损失曲线在同一张图上
def plot_comparison_training_history(history_original, history_augmented):
    # 准确率对比图
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history_original.history['accuracy'], label='原始数据集 - 训练准确率', linestyle='--', marker='o')
    plt.plot(history_original.history['val_accuracy'], label='原始数据集 - 验证准确率', linestyle='--', marker='o')
    plt.plot(history_augmented.history['accuracy'], label='旋转增强数据集 - 训练准确率', linestyle='-', marker='s')
    plt.plot(history_augmented.history['val_accuracy'], label='旋转增强数据集 - 验证准确率', linestyle='-', marker='s')
    plt.title('原始数据集与旋转增强数据集的准确率对比')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # 损失对比图
    plt.subplot(1, 2, 2)
    plt.plot(history_original.history['loss'], label='原始数据集 - 训练损失', linestyle='--', marker='o')
    plt.plot(history_original.history['val_loss'], label='原始数据集 - 验证损失', linestyle='--', marker='o')
    plt.plot(history_augmented.history['loss'], label='旋转增强数据集 - 训练损失', linestyle='-', marker='s')
    plt.plot(history_augmented.history['val_loss'], label='旋转增强数据集 - 验证损失', linestyle='-', marker='s')
    plt.title('原始数据集与旋转增强数据集的损失对比')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
# 绘制两个数据集的训练历史对比
plot_comparison_training_history(history_original, history_augmented)
# 混淆矩阵的可视化
def plot_confusion_matrix(model, X_test, y_test, title):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'{title} - 混淆矩阵')
    plt.show()

# 绘制混淆矩阵
plot_confusion_matrix(cnn_model_original, X_test_original, y_test_original, '原始数据集')
plot_confusion_matrix(cnn_model_augmented, X_test, y_test, '旋转增强数据集')

# 可视化准确率和损失对比条形图
def plot_comparison(accuracy_original, accuracy_augmented, loss_original, loss_augmented):
    labels = ['原始数据集', '旋转增强数据集']
    accuracy = [accuracy_original, accuracy_augmented]
    loss = [loss_original, loss_augmented]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(x - width / 2, accuracy, width, label='准确率', color='blue')
    ax1.set_xlabel('数据集类型')
    ax1.set_ylabel('准确率')
    ax1.set_title('不同数据集上的准确率和损失对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, loss, width, label='损失', color='orange')
    ax2.set_ylabel('损失')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.show()







