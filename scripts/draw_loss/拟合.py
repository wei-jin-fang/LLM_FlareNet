import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window_size):
    """ 计算移动平均，用于平滑曲线 """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 定义要绘制的行范围（基于 0 索引，不包括表头）
# 0 是除了表头的第一行，就是第0轮次
start_row = 1  # 起始行号
end_row = 50+1   # 结束行号（不包含此行）

result_dir = "Onefit"
for _modelType in ["Onefitall_16", "Onefitall_17", "Onefitall_18"]:
    # 文件路径
    train_loss_path = fr"../../weight/{_modelType}/{_modelType}_train_loss.csv"
    validation_loss_path = fr"../../weight/{_modelType}/{_modelType}_validation_loss.csv"

    # 读取CSV文件，跳过第一行表头
    train_loss_data = pd.read_csv(train_loss_path, header=None, skiprows=1)
    validation_loss_data = pd.read_csv(validation_loss_path, header=None, skiprows=1)

    # 选择指定行范围（基于 0 索引）
    train_loss_data = train_loss_data.iloc[start_row:end_row]
    validation_loss_data = validation_loss_data.iloc[start_row:end_row]

    # 绘制并保存训练损失曲线
    plt.figure(figsize=(10, 6))
    window_size = 2  # 窗口大小，根据需要调整

    for i in range(train_loss_data.shape[1]):
        smooth_data = moving_average(train_loss_data[i].dropna(), window_size)
        plt.plot(smooth_data, label=f'Curve {i}')
    plt.title(f"{_modelType} Training Loss Curves")
    plt.xlabel("Epochs")
    plt.xlim(0,50)
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"./{result_dir}/{_modelType}_training_loss_curves_rows_{start_row}_to_{end_row}.png")

    # 绘制并保存验证损失曲线
    plt.figure(figsize=(10, 6))
    for i in range(validation_loss_data.shape[1]):
        smooth_data = moving_average(validation_loss_data[i].dropna(), window_size)
        plt.plot(smooth_data, label=f'Curve {i}')
    plt.title(f"{_modelType} Validation Loss Curves")
    plt.xlabel("Epochs")
    plt.xlim(0,50)
    plt.ylabel("Loss")
    # plt.ylim(4, 18)  # 设置 y 轴范围，可以调整
    plt.xlim(0,50)
    plt.legend()
    plt.savefig(f"./{result_dir}/{_modelType}_validation_loss_curves_rows_{start_row}_to_{end_row}.png")
