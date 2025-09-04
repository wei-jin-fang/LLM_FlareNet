import pandas as pd
import matplotlib.pyplot as plt

# 设置全局字体和大小
plt.rcParams.update({
    'figure.figsize': (12, 8),  # 调整整体图像大小
    'axes.titlesize': 20,  # 标题字体大小
    'axes.labelsize': 18,  # x 轴和 y 轴标签字体大小
    'xtick.labelsize': 16,  # x 轴刻度字体大小
    'ytick.labelsize': 16,  # y 轴刻度字体大小
    'legend.fontsize': 10,  # 图例字体大小
    'lines.linewidth': 2,  # 线条宽度
})

# 定义要绘制的行范围（基于 0 索引，不包括表头）
# 0 是除了表头的第一行，就是第0轮次
start_row = 1  # 起始行号
end_row = 50+1   # 结束行号（不包含此行）

result_dir = "Onefit"
for _modelType in ["Onefitall_16", "Onefitall_17", "Onefitall_18"]:
    # 文件路径
    train_loss_path = fr"../../weight/{_modelType}/{_modelType}_train_loss.csv"
    validation_loss_path = fr"../../weight/{_modelType}/{_modelType}_validation_loss.csv"

    # 读取CSV文件，跳过第一行表头，并选择指定行范围
    train_loss_data = pd.read_csv(train_loss_path, header=None, skiprows=1)
    validation_loss_data = pd.read_csv(validation_loss_path, header=None, skiprows=1)


    # 选择指定行范围（基于 0 索引）
    train_loss_data = train_loss_data.iloc[start_row:end_row]
    validation_loss_data = validation_loss_data.iloc[start_row:end_row]
    # 绘制并保存训练损失曲线
    plt.figure(figsize=(10, 6))
    for i in range(train_loss_data.shape[1]):
        plt.plot(train_loss_data[i], label=f'Dataset {i}')
    plt.title(f"training loss curves ({_modelType})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.ylim(0, 15)  # 可根据需要调整 y 轴范围
    plt.xlim(0,50)
    plt.legend()
    plt.savefig(f"./{result_dir}/{_modelType}_training_loss_curves_rows_{start_row}_to_{end_row}.png")
    plt.close()

    # 绘制并保存验证损失曲线
    plt.figure(figsize=(10, 6))
    for i in range(validation_loss_data.shape[1]):
        plt.plot(validation_loss_data[i], label=f'Dataset {i}')
    plt.title(f"validation loss curves ({_modelType})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.ylim(0.3, 0.6)  # 设置 y 轴范围
    plt.xlim(0,50)
    plt.legend()
    plt.savefig(f"./{result_dir}/{_modelType}_validation_loss_curves_rows_{start_row}_to_{end_row}.png")
    plt.close()