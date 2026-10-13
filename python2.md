# 作业1、3：Ridge、Lasso、Elastic Net 与正则化

## 一、推导题：证明交叉设计下 $\beta_{Ridge}=\frac{1}{1+\lambda}\beta_{OLS}$

在岭回归中，目标函数为：

$$
\min_\beta \|y - X\beta\|^2 + \lambda \|\beta\|^2
$$

其最优解满足：

$$
\beta_{Ridge} = (X^T X + \lambda I)^{-1} X^T y
$$

若设计矩阵满足交叉设计（orthogonal design）条件，即：

$$
X^T X = I
$$

则代入上式可得：

$$
\beta_{Ridge} = (I + \lambda I)^{-1} X^T y
= (1 + \lambda)^{-1} I X^T y
= \frac{1}{1 + \lambda} X^T y
$$

而 OLS 的闭式解为：

$$
\beta_{OLS} = (X^T X)^{-1} X^T y
$$

在交叉设计下，$X^T X = I$，因此：

$$
\beta_{OLS} = I^{-1} X^T y = X^T y
$$

于是：

$$
\beta_{Ridge} = \frac{1}{1 + \lambda} \beta_{OLS}
$$

证毕。

说明：

- 这说明在特征互不相关、设计矩阵标准化后，岭回归相当于对 OLS 系数做了统一缩放。
- 当 $\lambda > 0$ 时，系数被压缩，$\lambda$ 越大，压缩越明显。
- 这也是岭回归“收缩系数、稳定估计”的核心思想。

## 二、证明题：证明当 $\lambda > 0$ 时，$X^T X + \lambda I$ 正定

要证明矩阵

$$
A = X^T X + \lambda I
$$

是正定矩阵，即对任意非零向量 $z$，都有：

$$
z^T A z > 0
$$

因为：

$$
z^T (X^T X + \lambda I) z
= z^T X^T X z + \lambda z^T z
= \|Xz\|^2 + \lambda \|z\|^2
$$

其中：

- $\|Xz\|^2 \ge 0$
- $\lambda \|z\|^2 > 0$（因为 $\lambda > 0$ 且 $z \ne 0$）

所以：

$$
z^T (X^T X + \lambda I) z > 0
$$

这说明矩阵 $X^T X + \lambda I$ 是正定矩阵。

结论：

- 当 $\lambda > 0$ 时，$X^T X + \lambda I$ 可逆；
- 也就是说，岭回归的闭式解存在且唯一；
- 这也是为什么岭回归能够避免 OLS 在特征共线性严重时出现病态问题。

## 三、分析题：为什么做正则化前必须做 Z-score 标准化？不做的后果是什么？

在 Ridge、Lasso、Elastic Net 中，正则化项通常是：

$$
\lambda \|\beta\|^2
\quad \text{或} \quad
\lambda_1 \|\beta\|_1 + \lambda_2 \|\beta\|^2
$$

其中，惩罚项对各个特征的系数统一施加约束。如果特征没有进行标准化，那么不同特征的量纲和尺度差异很大，导致正则化效果不公平。

### 1. 为什么要做 Z-score 标准化？

对每个特征 $x_j$ 进行标准化：

$$
z_j = \frac{x_j - \mu_j}{\sigma_j}
$$

这样处理后：

- 每个特征均值为 0
- 方差为 1
- 不同特征处于同一数量级上

此时正则项的惩罚对所有特征更“公平”，也就是每个特征的系数在同一尺度上被收缩，不会因为数值单位不同而被不合理地放大或缩小。

### 2. 不做标准化的后果

若不做标准化，可能出现以下问题：

- 大尺度特征会被“惩罚得更严重”；
- 小尺度特征可能被相对放大，导致模型对它们过度偏好；
- 正则化项无法真正反映特征重要性，而是被数值规模“误导”；
- 在 Ridge 中，系数收缩会严重受特征尺度影响；
- 在 Lasso 中，某些重要但数值较小的特征可能被压成 0，导致特征选择不合理；
- 模型难以比较不同特征的贡献，超参数 $\lambda$ 的选择也会失真。

因此，做 Z-score 标准化是正则化稳定和有效的关键前提之一。

更直观地说：

正则化不是“只惩罚系数大小”，而是“惩罚系数在数值尺度上的大小”；如果特征范围不同，惩罚项就不再公平。标准化之后，惩罚项对于所有特征都在同一尺度上起作用，模型更稳定。

## 四、实战题：用 Hitter / Boston 房价数据对比 Ridge、Lasso、Elastic Net

### 任务要求

- 使用 Hitter 或 Boston 房价数据
- 分别训练 Ridge、Lasso、Elastic Net
- 做数据划分、特征标准化、交叉验证调参
- 对比模型性能和系数稀疏性

### 1. 数据准备

以 Boston 房价数据为例（也可以使用 Hitter 数据集），代码思路如下：

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# 加载数据
X, y = load_boston(return_X_y=True)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 2. 模型构建

分别建立三个模型：

```python
ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge())
])

lasso = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Lasso(max_iter=10000))
])

enet = Pipeline([
    ("scaler", StandardScaler()),
    ("model", ElasticNet(max_iter=10000))
])
```

### 3. 交叉验证调参

对参数进行网格搜索：

```python
param_grid_ridge = {"model__alpha": [0.01, 0.1, 1, 10, 100]}
param_grid_lasso = {"model__alpha": [0.001, 0.01, 0.1, 1, 10]}
param_grid_enet = {
    "model__alpha": [0.001, 0.01, 0.1, 1, 10],
    "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
}

# Ridge
grid_ridge = GridSearchCV(ridge, param_grid_ridge, cv=5)
grid_ridge.fit(X_train, y_train)

# Lasso
grid_lasso = GridSearchCV(lasso, param_grid_lasso, cv=5)
grid_lasso.fit(X_train, y_train)

# Elastic Net
grid_enet = GridSearchCV(enet, param_grid_enet, cv=5)
grid_enet.fit(X_train, y_train)
```

### 4. 模型评价

对比训练后的测试集性能：

```python
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

ridge_mse, ridge_r2 = evaluate(grid_ridge.best_estimator_, X_test, y_test)
lasso_mse, lasso_r2 = evaluate(grid_lasso.best_estimator_, X_test, y_test)
enet_mse, enet_r2 = evaluate(grid_enet.best_estimator_, X_test, y_test)

print("Ridge: MSE=..., R²=...")
print("Lasso: MSE=..., R²=...")
print("Elastic Net: MSE=..., R²=...")
```

### 5. 结果分析

通常来说：

- Ridge：在共线性较强时表现稳定，通常预测效果较稳，系数不会突然变成 0；
- Lasso：对特征选择更强，系数可能很多为 0，模型更稀疏；
- Elastic Net：兼具 Ridge 和 Lasso 的优点，适合高维特征和相关特征并存的情况。

### 6. 实验结论

可以写成如下表述：

> 在 Boston 房价数据上，我们分别训练了 Ridge、Lasso 和 Elastic Net 模型，并在相同的训练/测试划分和交叉验证条件下比较其预测性能。实验中先对特征进行 Z-score 标准化，再用网格搜索选择最优正则化参数。结果表明，Ridge 在共线性较强的数据上表现更稳定，Lasso 通过稀疏化得到更简洁的模型，而 Elastic Net 结合了两者优点，在特征相关性较强时往往具有更好的综合表现。

## 五、总结

本题核心在于理解三类回归模型的差异：

- Ridge：收缩系数，适合共线性问题；
- Lasso：稀疏特征选择，适合高维选择问题；
- Elastic Net：平衡两者，适合相关特征较多的场景。

同时，标准化是正则化前的必要步骤，因为正则项会受特征尺度影响，只有在统一尺度下，惩罚才是公平和合理的。

综上，正则化方法的本质是：在不损失较强预测能力的前提下，对模型参数进行约束，从而改善过拟合、处理共线性并提升稳定性。
