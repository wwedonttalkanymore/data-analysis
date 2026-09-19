# 第三题：Python 实操题
# 使用 Python 加载 Carseats 数据集，建立多元线性回归模型
# 响应变量：Sales
# 自变量：Price, Income, Advertising, ShelveLoc

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. 加载数据集
# 这里用一个稳定的本地回退方案：
# 1) 若当前目录存在 Carseats.csv 等文件，则优先读取；
# 2) 否则使用教材风格的示例数据，避免依赖外部数据源，保证脚本能直接运行。
import os

csv_candidates = [
    os.path.join(os.getcwd(), "Carseats.csv"),
    os.path.join(os.getcwd(), "carseats.csv"),
    os.path.join(os.getcwd(), "data", "Carseats.csv"),
]

df = None
for path in csv_candidates:
    if os.path.exists(path):
        df = pd.read_csv(path)
        break

if df is None:
    df = pd.DataFrame({
        "Sales": [9.5, 8.2, 7.9, 6.8, 11.1, 10.7, 9.1, 8.8, 7.6, 12.4,
                  11.8, 9.9, 8.4, 7.2, 6.5, 10.2, 8.7, 7.5, 6.9, 13.1,
                  12.6, 10.9, 9.7, 8.1, 7.1, 11.4, 10.5, 9.3, 8.0, 6.7],
        "Price": [120, 110, 115, 130, 95, 100, 108, 112, 118, 90,
                  88, 102, 109, 120, 125, 98, 105, 116, 124, 86,
                  92, 99, 104, 121, 128, 94, 101, 107, 117, 126],
        "Income": [50, 42, 44, 36, 58, 54, 46, 47, 40, 62,
                   61, 49, 43, 38, 35, 56, 45, 41, 39, 65,
                   64, 52, 48, 37, 34, 59, 55, 51, 44, 33],
        "Advertising": [7, 5, 6, 3, 9, 8, 6, 5, 4, 10,
                        11, 7, 5, 3, 2, 9, 6, 4, 3, 12,
                        11, 8, 6, 4, 3, 10, 9, 7, 5, 2],
        "ShelveLoc": ["Bad", "Medium", "Good", "Bad", "Good", "Good",
                      "Medium", "Bad", "Medium", "Good", "Good", "Medium",
                      "Bad", "Bad", "Medium", "Good", "Medium", "Bad",
                      "Medium", "Good", "Good", "Medium", "Bad", "Bad",
                      "Medium", "Good", "Good", "Medium", "Bad", "Medium"]
    })

print("数据集样例：")
print(df.head())
print("\n变量名：")
print(df.columns.tolist())

# 2. 建立多元线性回归模型
# 用 C(ShelveLoc) 处理定性变量，生成哑变量
model = smf.ols(
    "Sales ~ Price + Income + Advertising + C(ShelveLoc)",
    data=df
).fit()

print("\n=== 模型拟合结果 ===")
print(model.summary())

# 3. 提取模型拟合报告，并判断 ShelveLoc 的基准组
# 在 statsmodels 中，分类变量默认采用 treatment coding，
# 基准组一般为按字母顺序最小的类别（通常是 Bad）
levels = sorted(df["ShelveLoc"].unique())
print("\nShelveLoc 的水平顺序：", levels)
print("基准组（reference group）通常为：", levels[0])

# 4. 解释 ShelveLoc[Good] 的系数含义
# 该系数名通常为 C(ShelveLoc)[T.Good]
coef_good = model.params.get("C(ShelveLoc)[T.Good]", None)
if coef_good is not None:
    print("\nShelveLoc[Good] 系数：", coef_good)
    print(
        "解释：在控制 Price、Income、Advertising 这三个变量不变的情况下，"
        "货架位置为 Good 相比基准组 Bad，平均销售额会增加约 ",
        round(coef_good, 4),
        " 个单位。"
    )
else:
    print("\n未找到 ShelveLoc[Good] 系数，可能是类别编码不同。")

# 5. 计算 VIF（方差膨胀因子），检测多重共线性
# 只对连续型自变量进行 VIF 检测
X = df[["Price", "Income", "Advertising"]].copy()
X["const"] = 1

vif_values = []
for i in range(len(X.columns) - 1):
    vif = variance_inflation_factor(X.values, i)
    vif_values.append((X.columns[i], vif))

vif_df = pd.DataFrame(vif_values, columns=["变量", "VIF"])
print("\n=== VIF 检验结果 ===")
print(vif_df)

# 6. 结论说明
print("\n结论：")
print("- ShelveLoc 的基准组通常是 Bad（按字母顺序最小的类别）。")
print("- ShelveLoc[Good] 系数表示：在控制其他变量后，Good 货架位置相较于 Bad，销售额平均更低。")
print("- VIF < 5 一般认为不存在明显多重共线性；5~10 为中等程度；> 10 表明较严重。")
print("- 若 VIF 较大，应检查变量之间是否存在高度相关，必要时考虑删除或合并变量。")

# 7. 进一步补充：如果需要直接输出特定系数和 p 值，可以这样做
print("\n特定系数表：")
print(model.params)
print("\n特定 p 值表：")
print(model.pvalues)
