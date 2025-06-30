file_name=r"D:\BaiduSyncdisk\01projs\02github_small\AlignedReID-master\log\751LSTHL\log_train.txt"

import re
from collections import defaultdict
import matplotlib.pyplot as plt

# 初始化字典以存储每个Epoch的损失值
epoch_losses = defaultdict(lambda: defaultdict(list))

# 打开并读取日志文件
with open(file_name, "r") as file:
    for line in file:
        # 使用正则表达式匹配行，并提取Epoch号和损失值
        match = re.search(r"Epoch: \[(\d+)\].*Loss (\d+\.\d+).*CLoss (\d+\.\d+).*GLoss (\d+\.\d+).*LLoss (\d+\.\d+)",
                          line)
        if match:
            epoch = int(match.group(1))
            epoch_losses[epoch]["Loss"].append(float(match.group(2)))
            epoch_losses[epoch]["CLoss"].append(float(match.group(3)))
            epoch_losses[epoch]["GLoss"].append(float(match.group(4)))
            epoch_losses[epoch]["LLoss"].append(float(match.group(5)))

# 准备文件以保存平均损失值
with open("average_losses_per_epoch.txt", "w") as avg_file:
    for epoch in sorted(epoch_losses.keys()):
        avg_file.write(f"Epoch: {epoch}\n")
        for loss_type in ["Loss", "CLoss", "GLoss", "LLoss"]:
            avg_loss = sum(epoch_losses[epoch][loss_type]) / len(epoch_losses[epoch][loss_type])
            avg_file.write(f"Average {loss_type}: {avg_loss:.4f}\n")
        avg_file.write("\n")

# 可视化
import matplotlib.pyplot as plt

# 初始化列表以存储每种损失的平均值
avg_loss_values = []
avg_c_loss_values = []
avg_g_loss_values = []
avg_l_loss_values = []
epochs = sorted(epoch_losses.keys())

# 计算每个Epoch的平均损失值
for epoch in epochs:
    avg_loss = sum(epoch_losses[epoch]["Loss"]) / len(epoch_losses[epoch]["Loss"])
    avg_c_loss = sum(epoch_losses[epoch]["CLoss"]) / len(epoch_losses[epoch]["CLoss"])
    avg_g_loss = sum(epoch_losses[epoch]["GLoss"]) / len(epoch_losses[epoch]["GLoss"])
    avg_l_loss = sum(epoch_losses[epoch]["LLoss"]) / len(epoch_losses[epoch]["LLoss"])

    avg_loss_values.append(avg_loss)
    avg_c_loss_values.append(avg_c_loss)
    avg_g_loss_values.append(avg_g_loss)
    avg_l_loss_values.append(avg_l_loss)

# 绘制图表
plt.figure(figsize=(10, 6),dpi=500)
plt.plot(epochs, avg_loss_values, label="Loss", marker='o')
plt.plot(epochs, avg_c_loss_values, label="CLoss", marker='s')
plt.plot(epochs, avg_g_loss_values, label="GLoss", marker='^')
plt.plot(epochs, avg_l_loss_values, label="LLoss", marker='x')

plt.title("Average Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.legend()
plt.grid(True)
plt.savefig("plotloss.png")
plt.show()

