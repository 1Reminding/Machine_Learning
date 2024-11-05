import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
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

# 定义回归模型参数范围
ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000]}
lasso_params = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
elastic_net_params = {'alpha': [0.001, 0.01, 0.1, 1], 'l1_ratio': [0.1, 0.2, 0.5, 0.8]}
svr_params = {'C': [0.1, 1, 10, 100], 'epsilon': [0.01, 0.1, 1]}
dt_params = {'max_depth': [3, 5, 10, 20], 'min_samples_split': [2, 5, 10]}
rf_params = {'n_estimators': [100, 200], 'max_depth': [5, 10, 20]}
gb_params = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1, 0.5], 'max_depth': [3, 5, 10]}

# 通用的网格搜索函数
def perform_grid_search(model, params, X_train, y_train):
    grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

# Ridge 回归
ridge_model, ridge_best_params = perform_grid_search(Ridge(), ridge_params, X_train_poly, y_train)
ridge_mse = mean_squared_error(y_test, ridge_model.predict(X_test_poly))
print(f"Ridge 最优参数: {ridge_best_params}, 测试集 MSE: {ridge_mse}")

# Lasso 回归
lasso_model, lasso_best_params = perform_grid_search(Lasso(), lasso_params, X_train_poly, y_train)
lasso_mse = mean_squared_error(y_test, lasso_model.predict(X_test_poly))
print(f"Lasso 最优参数: {lasso_best_params}, 测试集 MSE: {lasso_mse}")

# ElasticNet 回归
elastic_net_model, elastic_net_best_params = perform_grid_search(ElasticNet(), elastic_net_params, X_train_poly, y_train)
elastic_net_mse = mean_squared_error(y_test, elastic_net_model.predict(X_test_poly))
print(f"ElasticNet 最优参数: {elastic_net_best_params}, 测试集 MSE: {elastic_net_mse}")

# 支持向量回归 (SVR)
svr_model, svr_best_params = perform_grid_search(SVR(), svr_params, X_train_poly, y_train)
svr_mse = mean_squared_error(y_test, svr_model.predict(X_test_poly))
print(f"SVR 最优参数: {svr_best_params}, 测试集 MSE: {svr_mse}")

# 决策树回归 (Decision Tree)
dt_model, dt_best_params = perform_grid_search(DecisionTreeRegressor(), dt_params, X_train_poly, y_train)
dt_mse = mean_squared_error(y_test, dt_model.predict(X_test_poly))
print(f"决策树最优参数: {dt_best_params}, 测试集 MSE: {dt_mse}")

# 随机森林回归 (Random Forest)
rf_model, rf_best_params = perform_grid_search(RandomForestRegressor(), rf_params, X_train_poly, y_train)
rf_mse = mean_squared_error(y_test, rf_model.predict(X_test_poly))
print(f"随机森林最优参数: {rf_best_params}, 测试集 MSE: {rf_mse}")

# 梯度提升树回归 (Gradient Boosting)
gb_model, gb_best_params = perform_grid_search(GradientBoostingRegressor(), gb_params, X_train_poly, y_train)
gb_mse = mean_squared_error(y_test, gb_model.predict(X_test_poly))
print(f"梯度提升树最优参数: {gb_best_params}, 测试集 MSE: {gb_mse}")