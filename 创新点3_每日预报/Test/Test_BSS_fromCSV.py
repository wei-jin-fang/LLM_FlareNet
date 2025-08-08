import numpy as np
import pandas as pd

from tools import BS_BSS_score, Metric, truncate


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
    return metrics["BSS"]
if __name__ == '__main__':
    onebss_dan=getBestTss_BSSMetricsFromFile("../对比ccmc_单/对齐CSV概率与complete_BSS.csv",0.5,"one")[1]
    MPlusbss_dan=getBestTss_BSSMetricsFromFile("../对比ccmc_单/对齐CSV概率与complete_BSS.csv",0.5,"MPlus")[1]
    print(onebss_dan,MPlusbss_dan)

    onebss_duo = getBestTss_BSSMetricsFromFile("../对比ccmc_多/对齐CSV概率与complete_BSS.csv", 0.5, "one")[1]
    MPlusbss_duo = getBestTss_BSSMetricsFromFile("../对比ccmc_多/对齐CSV概率与complete_BSS.csv", 0.5, "MPlus")[1]
    print(onebss_duo, MPlusbss_duo)

    onebss_mix = getBestTss_BSSMetricsFromFile("../对比ccmc_混合/对齐CSV概率与complete_BSS.csv", 0.5, "one")[1]
    MPlusbss_mix = getBestTss_BSSMetricsFromFile("../对比ccmc_混合/对齐CSV概率与complete_BSS.csv", 0.5, "MPlus")[1]
    print(onebss_mix, MPlusbss_mix)