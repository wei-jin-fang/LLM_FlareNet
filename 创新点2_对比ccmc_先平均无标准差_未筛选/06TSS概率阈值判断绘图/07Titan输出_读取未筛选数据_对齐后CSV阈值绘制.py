import matplotlib
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
# 设置概率阈值
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

from 创新点整理版本.创新点1_未筛选.tools import Metric


def getTssFromFile(path):
    # 遍历目录中的所有CSV文件
    data_TSS = []
    # for i in range(10):
        # 存储结果的列表
    true_labels = []
    predicted_labels = []
    # 读取CSV文件
    df = pd.read_csv(path,encoding='ISO-8859-1')

    # 获取真实标签
    df['y_true'] = df['y_true'].astype(int)
    # 获取真实标签
    true_labels.extend(df['y_true'].tolist())
    # 计算预测标签
    predicted = (df['one'] > threshold).astype(int)  # 如果'one'列的值大于阈值，则预测为1，否则为0
    predicted_labels.extend(predicted.tolist())

    # 你现在有两个列表：true_labels 和 predicted_labels，分别包含所有CSV文件中的真实标签和预测标签

    metric = Metric(true_labels, predicted_labels)
    Tss = metric.TSS()[0]
    data_TSS.append(Tss)
    data_TSS = np.array(data_TSS)
    TSS_mean = data_TSS.mean(axis=0)
    TSS_std = data_TSS.std(axis=0)
    return TSS_mean


if __name__ == '__main__':
    for _modelType in ["LLM_VIT"]:
        # 设置步长
        step_size = 0.05
        # 初始化起始值
        start = 0.05
        model_type=_modelType
        # 使用for循环从0开始，增加至1（包含1）
        step_values = []
        for i in range(int(1 / step_size) + 1):  # +1 是为了确保1被包括进去
            step_value = start + i * step_size
            if step_value > 0.96:
                break  # 避免由于浮点精度问题超过1的情况
            step_values.append(step_value)
        # 遍历这些步长
        tssList=[]
        for step in step_values:
            threshold = step
            # 指定包含CSV文件的目录
            thisThresholdTss = getTssFromFile("Titan_combined_compare_output -对应ccmc个数.csv")
            print(thisThresholdTss)
            tssList.append(thisThresholdTss)

        # 创建图表
        plt.figure(figsize=(10, 5))  # 可以调整图表大小
        plt.plot(step_values, tssList, marker='o')  # 使用圆点标记每个数据点

        # 设置图表的标题和轴标签
        plt.title('LLM_VIT_TSS vs Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('TSS')

        # 设置横纵坐标的显示范围
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # 设置X轴和Y轴的刻度间隔为0.1
        ax = plt.gca()  # 获取当前轴对象
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))  # 设置X轴主刻度间隔
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))  # 设置Y轴主刻度间隔

        # 显示网格
        plt.grid(True)

        # 找到TSS值最大的点，并标记为红色，同时显示横纵坐标
        max_index = np.argmax(tssList)  # 获取最大TSS值的索引
        max_tss = tssList[max_index]  # 获取最大TSS值
        max_threshold = step_values[max_index]  # 获取对应的阈值
        plt.plot(max_threshold, max_tss, 'ro')  # 将最大的点标记为红色
        plt.text(max_threshold, max_tss, f'({max_threshold:.2f}, {max_tss:.2f})', ha='right', va='bottom')
        # 显示图表
        # plt.show()
        # 保存图表到文件
        plt.savefig(fr'compare_{model_type}_TSS_vs_Threshold_42个对齐.png', dpi=300)  # 指定分辨率为300 DPI


