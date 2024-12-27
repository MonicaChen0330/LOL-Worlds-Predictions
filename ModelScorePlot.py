import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 讀取 CSV 檔案
file_path = "results.csv"
df = pd.read_csv(file_path)

# 設定繪圖的欄位
accuracy_columns = ["A_wins_acc", "A_firstBlood_acc", "A_firstTowerKill_acc"]
roc_auc_columns = ["A_wins_roc", "A_firstBlood_roc", "A_firstTowerKill_roc"]

# 使用 Seaborn 的調色板
palette = sns.color_palette("coolwarm", len(accuracy_columns))  # 柔和的調色板
colors = [palette[i] for i in range(len(accuracy_columns))]  # 提取對應的顏色

# 繪製 Accuracy 的柱狀圖
plt.figure(figsize=(16, 12))
ax = df.plot(
    x="Model",
    y=accuracy_columns,
    kind="bar",
    figsize=(12, 8),
    title="Accuracy Comparison by Model",
    xlabel="Model",
    ylabel="Accuracy",
    rot=0,
    color=colors  # 使用 Seaborn 調色板的顏色
)

# 在每個 bar 上方顯示數值
for container in ax.containers:
    ax.bar_label(container, fmt='%.4f', padding=3, fontsize=12, color="black")  # 數值保留到小數點第 4 位

plt.legend(title="Metrics")
plt.tight_layout()
plt.savefig("accuracy_comparison.png")
plt.show()

# 繪製 ROC-AUC 的柱狀圖
plt.figure(figsize=(16, 12))
ax = df.plot(
    x="Model",
    y=roc_auc_columns,
    kind="bar",
    figsize=(12, 8),
    title="ROC-AUC Comparison by Model",
    xlabel="Model",
    ylabel="ROC-AUC",
    rot=0,
    color=colors  # 使用 Seaborn 調色板的顏色
)

# 在每個 bar 上方顯示數值
for container in ax.containers:
    ax.bar_label(container, fmt='%.4f', padding=3, fontsize=12, color="black")  # 數值保留到小數點第 4 位

plt.legend(title="Metrics")
plt.tight_layout()
plt.savefig("roc_auc_comparison.png")
plt.show()
