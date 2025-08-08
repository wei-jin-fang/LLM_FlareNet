import pandas as pd
from 创新点整理版本.创新点1_未筛选.tools import Metric

def getMetricsFromFile(path, threshold):
    # 读取CSV文件
    df = pd.read_csv(path, encoding='ISO-8859-1')
    df['y_true'] = df['y_true'].astype(int)  # 将真实标签转换为整数类型

    # 提取真实标签和预测标签
    true_labels = df['y_true'].tolist()
    predicted = (df['MPlus'] > threshold).astype(int)  # 根据阈值计算预测标签
    predicted_labels = predicted.tolist()

    # 初始化 Metric 类，用于计算各项指标
    metric = Metric(true_labels, predicted_labels)

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
def main():
    # 定义步长和起始阈值
    step_size = 0.05
    start = 0.05
    # 生成从 0.05 到 0.96 的步长列表
    step_values = [start + i * step_size for i in range(int(1 / step_size) + 1) if start + i * step_size <= 0.96]

    results = []  # 存储所有结果的列表

    for threshold in step_values:
        # 获取当前阈值下的所有指标
        metrics = getMetricsFromFile("Titan_combined_compare_output -对应ccmc个数.csv", threshold)
        for metric_name, metric_value in metrics.items():
            # 将每个指标的类别 [0] 和 [1] 值添加到结果列表中
            results.append([threshold, metric_name, metric_value[0], metric_value[1]])

    # 创建包含指定结构的 DataFrame
    df_results = pd.DataFrame(results, columns=["Threshold", "Metric Name/BS and BSS ", "Class 0 Value", "Class 1 Value"])
    print(df_results)
    # 保存为CSV文件
    output_path = "ccmc_metrics_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"结果已保存到 {output_path}")

if __name__ == '__main__':
    main()
