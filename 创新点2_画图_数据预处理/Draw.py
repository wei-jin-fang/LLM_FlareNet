# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
# import numpy as np  # 确保导入numpy
#
# # 设置matplotlib的字体配置，以正确显示中文
# rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体，例如使用黑体
# rcParams['axes.unicode_minus'] = False  # 正确显示负号
#
# def truncate(number, digits) -> float:
#     stepper = 10.0 ** digits
#     return np.trunc(stepper * number) / stepper
#
# def drawplot(isslect,directiondataset):
#     csv_paths = [
#         f"../创新点2_对比{directiondataset}_先平均无标准差_{isslect}/threshold_tss_data_one_LLM_VIT_TSS.csv",
#         f"../创新点2_对比{directiondataset}_先平均无标准差_{isslect}/threshold_tss_data_MPlus_LLM_VIT_TSS.csv",
#         f"../创新点2_对比{directiondataset}有标准差_{isslect}/threshold_tss_data_one_LLM_VIT_TSS.csv",
#         f"../创新点2_对比{directiondataset}有标准差_{isslect}/threshold_tss_data_MPlus_LLM_VIT_TSS.csv"
#     ]
#     comments = [
#         "无标准差_one",
#         "无标准差_MPlus",
#         "有标准差_one",
#         "有标准差_MPlus"
#     ]
#
#     dataframes = []
#
#     # 读取 CSV 文件
#     for path in csv_paths:
#         df = pd.read_csv(path)
#         dataframes.append(df)
#
#     # 绘制折线图
#     plt.figure(figsize=(10, 6))
#     for i, df in enumerate(dataframes):
#         plt.plot(df['Threshold'], df['TSS'], marker='o', label=f'{comments[i]}')
#         # 找到最大值点并标记
#         max_value_index = df['TSS'].idxmax()
#         max_threshold = df.at[max_value_index, 'Threshold']
#         max_tss = df.at[max_value_index, 'TSS']
#         plt.plot(max_threshold, max_tss, marker='^', markersize=10, color='red')
#         # 显示截断后的数值
#         plt.annotate(f'{truncate(max_tss, 3):.3f}', (max_threshold, max_tss), textcoords="offset points", xytext=(0, 10),
#                      ha='center')
#
#     plt.title(f'{isslect}_TSS vs Threshold Analysis')
#     plt.xlabel('Threshold')
#     plt.ylabel('TSS')
#     plt.legend()
#     plt.grid(True)
#
#     # 设置X轴和Y轴的刻度范围和间隔
#     plt.xticks(ticks=np.linspace(0, 1, num=21), labels=[f"{x:.2f}" for x in np.linspace(0, 1, num=21)])
#     plt.yticks(ticks=np.linspace(-0.1, 1, num=12))
#
#     # 保存图片到文件
#     plt.savefig(f"./{isslect}_{directiondataset}_TSS_vs_Threshold_Analysis.png", dpi=300)  # 指定保存路径和DPI
#
#     plt.show()
#
# if __name__ == '__main__':
#     # drawplot("未筛选","ccmc")
#     # drawplot("筛选","ccmc")
#     # drawplot("未筛选","sr")
#     # drawplot("筛选","sr")
#     pass
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

# 设置matplotlib的字体配置，以正确显示中文
rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体，例如使用黑体
rcParams['axes.unicode_minus'] = False  # 正确显示负号

plt.rcParams.update({
    'figure.figsize': (12, 8),  # 调整整体图像大小
    'axes.titlesize': 16,  # 标题字体大小
    'axes.labelsize': 18,  # x 轴和 y 轴标签字体大小
    'xtick.labelsize': 12.5,  # x 轴刻度字体大小
    'ytick.labelsize': 16,  # y 轴刻度字体大小
    'legend.fontsize': 12,  # 图例字体大小
    'lines.linewidth': 2,  # 线条宽度
})
def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return np.trunc(stepper * number) / stepper


def drawplot(isslect, directiondataset,offsets):
    csv_paths = [
        f"../创新点2_对比{directiondataset}_先平均无标准差_{isslect}/threshold_tss_data_one_LLM_VIT_TSS.csv",
        f"../创新点2_对比{directiondataset}_先平均无标准差_{isslect}/threshold_tss_data_MPlus_LLM_VIT_TSS.csv",
        f"../创新点2_对比{directiondataset}_有标准差_{isslect}/threshold_tss_data_one_LLM_VIT_TSS.csv",
        # f"../创新点2_对比{directiondataset}_有标准差_{isslect}/threshold_tss_data_MPlus_LLM_VIT_TSS.csv"
    ]
    if directiondataset == "CCMC":
        indirectiondataset = "NASA/CCMC"
    else:
        indirectiondataset = "SolarFlareNet"
    if isslect == "未筛选":
        inisslect = "mixed"
    else:
        inisslect = "single"
    comments = [
        f"LLMVITwithout",
        f"{indirectiondataset}",
        f"LLMVITwith",
        # f"{directiondataset}with",
    ]
    comments = [
        f"LLMVIT-Ⅰ",
        f"{indirectiondataset}",
        f"LLMVIT-Ⅱ",
        # f"{directiondataset}with",
    ]

    # 定义偏移量列表，每个元组表示一个 (offset_x, offset_y) 配置


    dataframes = []

    # 读取 CSV 文件
    for path in csv_paths:
        df = pd.read_csv(path)
        dataframes.append(df)

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    for i, df in enumerate(dataframes):
        plt.plot(df['Threshold'], df['TSS'], marker='o', label=f'{comments[i]}')

        # 找到最大值点并标记
        max_value_index = df['TSS'].idxmax()
        max_threshold = df.at[max_value_index, 'Threshold']
        max_tss = df.at[max_value_index, 'TSS']

        # 使用 offsets 列表中的偏移量
        offset_x, offset_y = offsets[i]
        plt.plot(max_threshold, max_tss, marker='^', markersize=10, color='red')

        # 显示截断后的数值并应用偏移量
        plt.annotate(
            f'{truncate(max_tss, 3):.3f}',
            (max_threshold, max_tss),
            textcoords="offset points",
            xytext=(offset_x, offset_y),
            ha='center'
        )
    if isslect == "未筛选":
        ARin = "ARs"
    else:
        ARin = "AR"
    plt.title(f'TSS curves of the LLMVIT and {indirectiondataset} on {inisslect} {ARin} in selected mode')
    plt.xlabel('Threshold')
    plt.ylabel('TSS')
    plt.legend()
    plt.grid(True)

    # 设置X轴和Y轴的刻度范围和间隔
    plt.xticks(ticks=np.linspace(0, 1, num=21), labels=[f"{x:.2f}" for x in np.linspace(0, 1, num=21)])
    plt.yticks(ticks=np.linspace(-0.1, 1, num=12))

    # 保存图片到文件
    plt.savefig(f"./{isslect}_{directiondataset}_TSS_vs_Threshold_Analysis.png", dpi=300)  # 指定保存路径和DPI

    # plt.show()


if __name__ == '__main__':
    # 24 相同
    drawplot("未筛选", "CCMC",[(0, 10), (0, 10), (0, -20), (0, 10)])
    drawplot("筛选", "CCMC",[(0, 10), (0, 10), (0, -20), (0, 10)])
    drawplot("未筛选", "SR",[(0, 10), (0, 10), (0, -20), (0, 10)])
    drawplot("筛选", "SR",[(0, 5), (0, 10), (0, -20), (0, 10)])
    pass

