
import matplotlib.pyplot as plt

acc_no_ft = [89.546783625731, 79.09356725146199, 69.88304093567251, 53.36257309941521, 47.953216374269005]
# Finetuning (蓝色方块)
acc_ft =  [90.78947368421052, 88.74269005847954, 87.42690058479532, 85.30701754385964, 84.64912280701755]


# 数据定义
thresholds = [0.1, 0.2, 0.4, 0.6, 0.8]

# No Finetuning 数据（红色圆点）
runtime_no_ft = [55, 49, 42, 37, 31]    # CPU Runtime (ms)


# Finetuning 数据（蓝色方块）
runtime_ft   = [53, 51, 39, 35, 29]      # CPU Runtime (ms)     # Accuracy (%)

plt.figure(figsize=(8, 6))

# 绘制 No Finetuning 数据，红色圆点
plt.scatter(runtime_no_ft, acc_no_ft, color='red', marker='o', s=80, label='No Finetuning')
# 绘制 Finetuning 数据，蓝色方块
plt.scatter(runtime_ft, acc_ft, color='blue', marker='s', s=80, label='Finetuning')

# 可选：为每个数据点添加阈值标注
for i, thr in enumerate(thresholds):
    plt.text(runtime_no_ft[i], acc_no_ft[i]-2, f"{thr}", color='red', ha='center', fontsize=9)
    plt.text(runtime_ft[i], acc_ft[i]+2, f"{thr}", color='blue', ha='center', fontsize=9)

plt.xlabel("MCU Runtime (ms)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs. Runtime on MCU")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
