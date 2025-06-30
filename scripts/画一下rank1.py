file_name=file_name=r"D:\BaiduSyncdisk\01projs\02github_small\AlignedReID-master\log\751LSTHL\log_train.txt"
import re
import matplotlib.pyplot as plt

# 初始化字典以存储每个Epoch的测试结果
test_results = {"mAP": [], "Rank-1": [], "Rank-5": [], "Rank-10": [], "Rank-20": []}
epochs = [i for i in range(10, 101, 10)]

with open(file_name, "r") as file:
    lines = file.readlines()
    for line in lines:
        mAP_match = re.search(r"mAP: (\d+\.\d+)%", line)
        rank1_match = re.search(r"Rank-1\s+:\s+(\d+\.\d+)%", line)
        rank5_match = re.search(r"Rank-5\s+:\s+(\d+\.\d+)%", line)
        rank10_match = re.search(r"Rank-10\s+:\s+(\d+\.\d+)%", line)
        rank20_match = re.search(r"Rank-20\s+:\s+(\d+\.\d+)%", line)
        if mAP_match:
            test_results["mAP"].append(float(mAP_match.group(1)))
        if rank1_match:
            test_results["Rank-1"].append(float(rank1_match.group(1)))
        if rank5_match:
            test_results["Rank-5"].append(float(rank5_match.group(1)))
        if rank10_match:
            test_results["Rank-10"].append(float(rank10_match.group(1)))
        if rank20_match:
            test_results["Rank-20"].append(float(rank20_match.group(1)))

# 准备绘制图表
plt.figure(figsize=(10, 6),dpi=500)

# 绘制mAP曲线
plt.plot(epochs, test_results["mAP"], label="mAP", marker='o')
# 绘制Rank-1曲线
plt.plot(epochs, test_results["Rank-1"], label="Rank-1", marker='s')
# 绘制Rank-5曲线
plt.plot(epochs, test_results["Rank-5"], label="Rank-5", marker='^')
# 绘制Rank-10曲线
plt.plot(epochs, test_results["Rank-10"], label="Rank-10", marker='x')
# 绘制Rank-20曲线
plt.plot(epochs, test_results["Rank-20"], label="Rank-20", marker='d')

plt.title("Test Results Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.savefig("pltrank1.png")
plt.show()
