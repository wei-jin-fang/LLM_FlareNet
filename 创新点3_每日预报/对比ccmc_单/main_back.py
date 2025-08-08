import pandas as pd

def addcol(path):
    # 读取没有表头的 CSV 文件
    df = pd.read_csv(path, header=None)



    # 按照图片中的内容设置表头
    df.columns = [
        "T_REC", "_modelTyp", "JianceType", "NOAA_ARS",
        "CV0_prob", "CV1_prob", "CV2_prob", "CV3_prob",
        "CV4_prob", "CV5_prob", "CV6_prob", "CV7_prob",
        "CV8_prob", "CV9_prob", "Nmbr", "one",  #one是为了取代cvproba
        "Class", "noaaID","predictday"
    ]


    # 删除 'new_noaaid' 列为空或 NaN 的行  new_noaaid为了方便后面代码对应改成noaaid
    df = df.dropna(subset=['noaaID'])

    # 进行筛选
    # 筛选 NOAA_ARS 字段仅包含一个数字的行
    df = df[df['NOAA_ARS'].apply(lambda x: len(x.split(',')) == 1)]

    # 根据 JianceType 和 NOAA_ARS 进行分割
    # 假设 JianceType 的值为 'TSS' 和 'BSS' 需要分割

    # 创建两个新的 DataFrame，一个包含 'TSS' 类型，另一个包含 'BSS' 类型
    df_tss = df[df['JianceType'] == 'TSS']
    df_bss = df[df['JianceType'] == 'BSS']

    # 将分割后的 DataFrame 分别保存为新的 CSV 文件
    df_tss.to_csv(f'result_forecast_TSS_cleaned.csv', index=False)
    df_bss.to_csv(f'result_forecast_BSS_cleaned.csv', index=False)




import csv
import os
import shutil
from astropy.io import fits
# 将数据写入 CSV 文件
import matplotlib
import numpy as np
import pandas as pd
import os

# 设置概率阈值
from matplotlib import pyplot as plt, ticker

from tools import Metric, BS_BSS_score, truncate

matplotlib.use('TkAgg')
import glob
import csv
import os
import shutil
from astropy.io import fits
# 将数据写入 CSV 文件
import matplotlib
import numpy as np
import pandas as pd
import os

# 设置概率阈值
from matplotlib import pyplot as plt, ticker

from tools import Metric, BS_BSS_score, truncate

matplotlib.use('TkAgg')
import glob
def duiqiOneandMPLus(JianceType):
    '''
    我现在有两个csv
    第一个是average_probabilities_path = rf"./result_forecast.csv_{JianceType}_cleaned.csv"
    第二个是SC_waitcompare.csv
    你需要拿出第一个csv的T_REC与noaaID列这两个字段和第二个表的prediction_date和noaaID列进行综合对比，
    只有某一行这两个字段均相等后，提取出第一个表这一行的Class字段数值和CV_average。提取出来第二个表的MPlus 字段，与他们共同的T_REC与noaaID拼成一行
    保存下来到一个新的csv
    '''
    average_probabilities_path = rf"./result_forecast_{JianceType}_cleaned.csv"
    df1 = pd.read_csv(average_probabilities_path)
    sc_waitcompare_path="../ccmc_waitcompare.csv"
    df2 = pd.read_csv(sc_waitcompare_path)


    # 确保字段名一致
    # 第一个表中的 T_REC 与 noaaID 和第二个表中的 prediction_date 与 noaaID 进行匹配
    df1 = df1[['T_REC', 'noaaID', 'Class', 'one']]  # 提取必要的列
    df2 = df2[['prediction_date', 'noaaID', 'MPlus_avg']]  # 提取必要的列

    # 为了进行合并，我们需要确保两个表中的日期字段名称一致
    df2.rename(columns={'prediction_date': 'T_REC'}, inplace=True)
    df2.rename(columns={'MPlus_avg': 'MPlus'}, inplace=True)

    # 合并两个数据框，匹配条件是 T_REC 和 noaaID
    merged_df = pd.merge(df1, df2, on=['T_REC', 'noaaID'], how='inner')

    output_path = f"对齐CSV概率与complete_{JianceType}.csv"
    # 将合并后的结果保存到新的 CSV 文件中
    merged_df.to_csv(f"对齐CSV概率与complete_{JianceType}.csv", index=False)
    # 重新读取保存的文件
    combined_df = pd.read_csv(output_path)

    # 根据 'Class' 字段添加 'y_true' 列
    combined_df['y_true'] = combined_df['Class'].apply(lambda x: 0 if x in ['N', 'C'] else 1)

    # 将包含 y_true 列的 DataFrame 再次保存到同一文件
    combined_df.to_csv(output_path, index=False)
    print(combined_df)
    print(f"文件已更新，添加 y_true 列，并保存至 '{output_path}'")


def getTssFromFile(path,threshold,ProName):
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
    predicted = (df[ProName] > threshold).astype(int)  # 如果'one'列的值大于阈值，则预测为1，否则为0
    predicted_labels.extend(predicted.tolist())

    # 你现在有两个列表：true_labels 和 predicted_labels，分别包含所有CSV文件中的真实标签和预测标签

    metric = Metric(true_labels, predicted_labels)
    Tss = metric.TSS()[0]
    data_TSS.append(Tss)
    data_TSS = np.array(data_TSS)
    TSS_mean = data_TSS.mean(axis=0)
    TSS_std = data_TSS.std(axis=0)
    return TSS_mean
def drawTSSPlot(Path,ProName,JinaceType):
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
            thisThresholdTss = getTssFromFile(Path,threshold,ProName)
            print(thisThresholdTss)
            tssList.append(    truncate(thisThresholdTss, 3))
        # 保存阈值和TSS数据到CSV文件
        csv_filename = fr'threshold_tss_data_{ProName}_{model_type}_{JinaceType}.csv'
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Threshold', 'TSS'])
            writer.writerows(zip(step_values, tssList))
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
        plt.text(max_threshold, max_tss, f'({truncate(max_threshold, 3):.3f}, {max_tss:.2f})', ha='right', va='bottom')
        # 显示图表
        # plt.show()
        # 保存图表到文件
        plt.savefig(fr'compare_{ProName}_{model_type}_{JinaceType}_vs_Threshold.png', dpi=300)  # 指定分辨率为300 DPI
        return max_threshold
def getMetricsFromFile(path, threshold,ProbName):
    # 读取CSV文件
    df = pd.read_csv(path, encoding='ISO-8859-1')
    df['y_true'] = df['y_true'].astype(int)  # 将真实标签转换为整数类型

    # 提取真实标签和预测标签
    true_labels = df['y_true'].tolist()
    predicted = (df[f'{ProbName}'] > threshold).astype(int)  # 根据阈值计算预测标签
    predicted_labels = predicted.tolist()

    # 初始化 Metric 类，用于计算各项指标
    metric = Metric(true_labels, predicted_labels)
    print(metric.Matrix())
    # 计算每个指标，确保每个指标为包含类别 [0] 和 [1] 的列表
    metrics = {
        "Accuracy": [metric.Accuracy()[0], metric.Accuracy()[1]],  # 准确率
        "Recall": [metric.Recall()[0], metric.Recall()[1]],        # 召回率
        "Precision": [metric.Precision()[0], metric.Precision()[1]],  # 精确率
        "TSS": [metric.TSS()[0], metric.TSS()[1]],                 # 真负率减去假正率
        # "BSS": [metric.BSS()[0], metric.BSS()[1]],                 # Brier技能得分
        "HSS": [metric.HSS()[0], metric.HSS()[1]],                 # Heidke技能得分
        "FAR": [metric.FAR()[0], metric.FAR()[1]],                 # 虚警率
        "FPR": [metric.FPR()[0], metric.FPR()[1]]                  # 假正率
    }

    return metrics
def getBestTss_BSSMetricsFromFile(path, threshold,ProbName):
    # 读取CSV文件
    df = pd.read_csv(path, encoding='ISO-8859-1')
    df['y_true'] = df['y_true'].astype(int)  # 将真实标签转换为整数类型

    # 提取真实标签和预测标签
    true_labels = df['y_true'].tolist()
    predicted = (df[f'{ProbName}'] > threshold).astype(int)  # 根据阈值计算预测标签
    predicted_labels = predicted.tolist()

    y_true = df['y_true'].tolist()
    y_prob = np.array(df[f'{ProbName}'].tolist())

    BS, BSS = BS_BSS_score(y_true, y_prob)
    # 初始化 Metric 类，用于计算各项指标
    metric = Metric(true_labels, predicted_labels)

    # 计算每个指标，确保每个指标为包含类别 [0] 和 [1] 的列表
    metrics = {
        "Accuracy": [metric.Accuracy()[0], metric.Accuracy()[1]],  # 准确率
        "Recall": [metric.Recall()[0], metric.Recall()[1]],        # 召回率
        "Precision": [metric.Precision()[0], metric.Precision()[1]],  # 精确率
        # "TSS": [metric.TSS()[0], metric.TSS()[1]],                 # 真负率减去假正率
        "BSS": [BS, BSS ],                 # Brier技能得分
        "HSS": [metric.HSS()[0], metric.HSS()[1]],                 # Heidke技能得分
        "FAR": [metric.FAR()[0], metric.FAR()[1]],                 # 虚警率
        "FPR": [metric.FPR()[0], metric.FPR()[1]]                  # 假正率
    }

    return metrics
def getBestTssMetrics(threshold,ProbName,JianceType):

    results = []  # 存储所有结果的列表
    results_excel = []  # 存储所有结果的列表
    for threshold in [threshold]:
        # 获取当前阈值下的所有指标
        metrics = getMetricsFromFile(rf"./对齐CSV概率与complete_{JianceType}.csv", threshold,ProbName)
        for metric_name, metric_value in metrics.items():
            # 将每个指标的类别 [0] 和 [1] 值添加到结果列表中
            results.append([threshold, metric_name,     truncate(metric_value[0], 3),     truncate(metric_value[1], 3)])
            results_excel.append([metric_name,     truncate(metric_value[0], 3),     truncate(metric_value[1], 3)])
    # 创建包含指定结构的 DataFrame
    df_results = pd.DataFrame(results,
                              columns=["Threshold", "Metric Name/BS and BSS ", "Class 0 Value", "Class 1 Value"])
    print(df_results)
    # 保存为CSV文件
    output_path = f"{ProbName}_{JianceType}_BestTSS_metrics_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"结果已保存到 {output_path}")

    # 创建包含指定结构的 DataFrame
    df_results = pd.DataFrame(results_excel, columns=["Metric", "负类", "正类"])
    # 保存为 Excel 文件
    output_excel_path = f"result_{ProbName}_{JianceType}.xlsx"
    df_results.to_excel(output_excel_path, index=False, engine='openpyxl')  # 使用 openpyxl 保存为 .xlsx 格式

def getBestTss_BSSMetrics(threshold,ProbName,JianceType):

    results = []  # 存储所有结果的列表
    results_excel=[]
    for threshold in [threshold]:
        # 获取当前阈值下的所有指标
        metrics = getBestTss_BSSMetricsFromFile(fr"./对齐CSV概率与complete_{JianceType}.csv", threshold,ProbName)
        for metric_name, metric_value in metrics.items():
            # 将每个指标的类别 [0] 和 [1] 值添加到结果列表中
            results.append([threshold, metric_name,     truncate(metric_value[0], 3),    truncate(metric_value[1], 3) ])
            results_excel.append([metric_name,     truncate(metric_value[0], 3),    truncate(metric_value[1], 3) ])
    # 创建包含指定结构的 DataFrame
    df_results = pd.DataFrame(results,
                              columns=["Threshold", "Metric Name/BS and BSS ", "Class 0 Value", "Class 1 Value"])
    print(df_results)
    # 保存为CSV文件
    output_path = f"{ProbName}_{JianceType}_BestTSS_metrics_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"结果已保存到 {output_path}")

    # 创建包含指定结构的 DataFrame
    df_results = pd.DataFrame(results_excel, columns=["Metric", "负类", "正类"])
    # 保存为 Excel 文件
    output_excel_path = f"result_{ProbName}_{JianceType}.xlsx"
    df_results.to_excel(output_excel_path, index=False, engine='openpyxl')  # 使用 openpyxl 保存为 .xlsx 格式


if __name__ == '__main__':
    addcol("../result_forecast.csv")
    print("=============================================================第一种先取概率One平均进行对齐========================================================")
    duiqiOneandMPLus("TSS")
    duiqiOneandMPLus("BSS")
    print("=============================================================绘图One与Mplus概率阈值图========================================================")
    max_one_index= drawTSSPlot("./对齐CSV概率与complete_TSS.csv","one","TSS")
    max_MPlus_index= drawTSSPlot("./对齐CSV概率与complete_TSS.csv","MPlus","TSS")
    BSSmax_one_index= drawTSSPlot("./对齐CSV概率与complete_BSS.csv","one","BSS")
    BSSmax_MPlus_index= drawTSSPlot("./对齐CSV概率与complete_BSS.csv","MPlus","BSS")
    print("=============================================================绘图One与Mplus最优TSS指标计算========================================================")
    getBestTssMetrics(max_one_index,"one","TSS")
    getBestTssMetrics(max_MPlus_index,"MPlus","TSS")
    getBestTss_BSSMetrics(BSSmax_one_index,"one","BSS")
    getBestTss_BSSMetrics(BSSmax_MPlus_index,"MPlus","BSS")
