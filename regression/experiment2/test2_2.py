import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# X, y 是特征和目标变量的数据集
file_path = 'winequality-white.csv'
data = pd.read_csv(file_path)
# 将数据集分为特征X和目标y
X = data.drop(columns=["quality"])
y = data["quality"]

# 读取数据并分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用 PolynomialFeatures 生成多项式特征
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# 构建 OLS 回归模型
X_train_poly_const = sm.add_constant(X_train_poly)
ols_model = sm.OLS(y_train, X_train_poly_const)
ols_result = ols_model.fit()

# 显示回归结果，包含 p 值、置信区间和显著性检验
print(ols_result.summary())

# 使用训练好的模型进行预测，并计算测试集上的 MSE
X_test_poly_const = sm.add_constant(X_test_poly)
y_pred = ols_result.predict(X_test_poly_const)
mse = mean_squared_error(y_test, y_pred)
print(f"OLS 多项式回归模型在测试集上的 MSE: {mse}")

# 分析模型解释能力 (R-squared) 和变量的显著性
r_squared = ols_result.rsquared
adj_r_squared = ols_result.rsquared_adj
print(f"模型的 R-squared: {r_squared}")
print(f"模型的 Adjusted R-squared: {adj_r_squared}")

# 提取各个特征的系数、p 值、t 值和置信区间
coefficients = ols_result.params
p_values = ols_result.pvalues
t_values = ols_result.tvalues
conf_int = ols_result.conf_int()

print(f"回归系数: {coefficients}")
print(f"p 值: {p_values}")
print(f"t 值: {t_values}")
print(f"95% 置信区间: {conf_int}")

# 判断变量显著性（一般p值 < 0.05 时认为变量具有统计显著性）
significant_vars = p_values[p_values < 0.05]
print(f"具有统计显著性的变量: {significant_vars}")
