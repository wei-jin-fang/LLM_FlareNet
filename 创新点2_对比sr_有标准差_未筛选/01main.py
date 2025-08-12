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

# from run_main_Test import mainTestdataFile
from tools import Metric, BS_BSS_score, truncate

matplotlib.use('TkAgg')
import glob
def check_and_replace_csv(folder_path, target_string, replacement_string, output_folder):
    # 确保输出文件夹存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        # 检查文件是否为CSV文件
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            # 读取CSV文件
            try:
                df = pd.read_csv(file_path)

                # 创建一个标志变量，检测是否需要替换字符串
                contains_target = df.astype(str).apply(lambda x: target_string in x.values).any()

                # 如果DataFrame中包含目标字符串，则进行替换
                if contains_target:
                    print(f"文件 '{file_name}' 包含指定字符串 '{target_string}'，正在进行替换...")
                    df = df.replace(target_string, replacement_string, regex=True)

                # 构建输出文件路径
                output_file_path = os.path.join(output_folder, file_name)

                # 保存处理后的CSV文件到输出文件夹
                df.to_csv(output_file_path, index=False)
                print(f"文件 '{file_name}' 已保存至 '{output_folder}'")

            except Exception as e:
                print(f"无法处理文件 '{file_name}'，错误: {e}")
def check_and_add_columns(folder_path,Columns):
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        # 检查文件是否为CSV文件
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            try:
                # 尝试读取文件的第一行以检查是否包含列名
                df = pd.read_csv(file_path)

                # 检查列名是否与预定义的列名一致
                if list(df.columns) != Columns:
                    print(f"文件 '{file_name}' 没有列名或列名不一致，正在添加列名...")

                    # 添加列名并保存
                    df.columns = Columns
                    df.to_csv(file_path, index=False)  # 保存添加列名后的文件
                    print(f"文件 '{file_name}' 列名已添加并保存")
                else:
                    print(f"文件 '{file_name}' 列名已经正确")

            except Exception as e:
                print(f"无法处理文件 '{file_name}'，错误: {e}")
def getTenFeature():
    # 设置基础目录，即"23_24SHARPImage_40"文件夹的路径
    base_directory = r'G:\本科\SHARP_23_24\23_24SHARPImage_40'

    # 用于存储结果的列表
    fits_files_info = []

    # 需要的 header 键
    required_keys = ['TOTUSJH', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'USFLUX', 'AREA_ACR', 'MEANPOT', 'R_VALUE',
                     'SHRGT45']

    # 遍历 N、C、M、X 四个子文件夹
    for category in ['N', 'C', 'M', 'X']:
        category_folder = os.path.join(base_directory, category)

        # 检查类别文件夹是否存在
        if os.path.exists(category_folder):
            # 遍历数字命名的文件夹
            for numbered_folder in os.listdir(category_folder):
                numbered_folder_path = os.path.join(category_folder, numbered_folder)

                # 确保它是一个文件夹
                if os.path.isdir(numbered_folder_path):
                    # 列出数字文件夹中的所有 .fits 文件
                    fits_files = [f for f in os.listdir(numbered_folder_path) if f.endswith('.fits')]

                    # 记录每个 .fits 文件的类别、数字编号和文件名
                    for fits_file in fits_files:
                        fits_file_path = os.path.join(numbered_folder_path, fits_file)

                        # 读取 .fits 文件的 header 信息
                        with fits.open(fits_file_path) as hdul:
                            header = hdul[-1].header  # 获取最后一个扩展的 header

                            # 只提取所需的键值对
                            header_info = {key: header.get(key, None) for key in required_keys}

                        # 将 header 信息添加到该文件的记录中
                        fits_files_info.append({
                            'CLASS': category,
                            'noaaID': numbered_folder,
                            'FITS文件': fits_file,
                            **header_info  # 将提取的 header 信息作为列
                        })

    # 将结果转换为 DataFrame 便于查看
    fits_files_df = pd.DataFrame(fits_files_info)

    output_csv_path = r'sr_fits提取_未筛选.csv'
    fits_files_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    # 重新读取 CSV 文件
    filtered_df = pd.read_csv(output_csv_path)

    # 删除 TOTUSJZ 列中没有数值的行
    filtered_df = filtered_df.dropna(subset=['TOTUSJZ'])

    # 将过滤后的数据再次写回 CSV 文件
    filtered_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"已删除 TOTUSJZ 列中没有数值的行，并重新写入 {output_csv_path}")
def globaldata():
    print("++++++++++++++++++++++++++++++++第一步进行数据清洗+++++++++++++++++++++++++++++++++++++++++++++")
    import os

    # 示例用法
    folder_path = r'../../Data_origin'
    output_folder = r'./clean之后数据'  # 当前py文件目录下的新文件夹，可以自定义文件夹名称
    target_string = '2.939.1'
    replacement_string = '2.939'  # 替换的字符串
    check_and_replace_csv(folder_path, target_string, replacement_string, output_folder)

    print("++++++++++++++++++++++++++++++++第二步骤添加列名字+++++++++++++++++++++++++++++++++++++++++++++")
    import os
    import pandas as pd

    # 预定义的列名
    Columns = ["NOAA_AR", "T_REC", "NOAA_NUM", "CLASS", "TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP",
               "USFLUX", "AREA_ACR", "MEANPOT", "R_VALUE", "SHRGT45"]



    # 示例用法
    folder_path = output_folder  # 替换为你的文件夹路径
    check_and_add_columns(folder_path,Columns)

    print("++++++++++++++++++++++++++++++++第三步合并指定csv+++++++++++++++++++++++++++++++++++++++++++++")
    import pandas as pd

    # 声明csv文件路径列表
    numdataset = 0
    csv_files = [rf'{output_folder}\{numdataset}Train.csv',
                 rf'{output_folder}\{numdataset}Val.csv',
                 rf'{output_folder}\{numdataset}Test.csv',
                 rf'{output_folder}\{numdataset + 1}Train.csv',
                 rf'{output_folder}\{numdataset + 1}Val.csv',
                 rf'{output_folder}\{numdataset + 1}Test.csv'

                 ]

    # 读取并合并csv文件
    df_list = [pd.read_csv(file) for file in csv_files]
    merged_df = pd.concat(df_list, axis=0, ignore_index=True)

    # 保存为新的csv文件
    merged_df.to_csv(rf'./merged{numdataset}_output.csv', index=False)
    print("合并完成")

    print("++++++++++++++++++++++++++++++++第四步归一化+++++++++++++++++++++++++++++++++++++++++++++")
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import os

    waitforStandfilepath = fr"./merged{numdataset}_output.csv"
    # 指定要标准化的列名
    Columns = ["NOAA_AR", "T_REC", "NOAA_NUM", "CLASS", "TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP",
               "USFLUX", "AREA_ACR", "MEANPOT", "R_VALUE", "SHRGT45"]

    # 指定标准化的列范围
    normalize_columns = Columns[4:14]  # 第5到第14列进行标准化
    train_data_ = pd.read_csv(waitforStandfilepath)
    train_data_norm = train_data_.copy()
    # 初始化 StandardScaler
    scaler = StandardScaler()
    # 对训练集的第5到第14列进行标准化计算（不转换）
    scaler.fit(train_data_norm[normalize_columns])

    # 输出标准化的均值和方差
    print("标准化的均值 (Mean):")
    print(scaler.mean_)
    print("\n标准化的方差 (Variance):")
    print(scaler.var_)

    # 读取新的CSV文件
    new_csv_path = "./sr_fits提取_未筛选.csv"  # 替换为新CSV文件的路径
    new_data = pd.read_csv(new_csv_path)

    # 使用之前的scaler对新数据中的指定列进行标准化
    new_data_normalized = new_data.copy()
    new_data_normalized[normalize_columns] = scaler.transform(new_data[normalize_columns])

    # 保存归一化后的数据到新的CSV文件
    output_csv_path = "../创新点2_对比sr_先平均无标准差_未筛选/scaler_sr_fits提取_未筛选.csv"  # 设置输出文件的路径
    new_data_normalized.to_csv(output_csv_path, index=False)

    # 确认消息
    print(f"归一化后的数据已经保存到 '{output_csv_path}'")

    os.remove(fr"./merged{numdataset}_output.csv")
    if os.path.exists("./clean之后数据"):
        shutil.rmtree("./clean之后数据")

def duiqiOneandMPLus(JianceType,datasetnumber):
    #       接下来是概率预报阈值对比
    file_paths = glob.glob(fr"./{JianceType}CSV/{JianceType}_LLM_VIT_{datasetnumber}.csv")
    # print(file_paths)

    cumulative_df = None
    print("++++++++++++++++++循环累加取平均值++++++++++++++++++++++++++++++++++")
    # Loop over each CSV file to sum up the probabilities
    for file_path in file_paths:
        df = pd.read_csv(file_path)

        # Initialize cumulative_df with the first file's structure
        if cumulative_df is None:
            cumulative_df = df[['zero', 'one']].copy()
        else:
            cumulative_df['zero'] += df['zero']
            cumulative_df['one'] += df['one']

    # Calculate the average by dividing by the number of files
    average_df = cumulative_df / len(file_paths)
    print("++++++++++++++++++ get叶180NOAAID_方便拼接csv++++++++++++++++++++++++++++++++++")
    # Step 2: Extract the NOAAID values every 40th row from the original CSV
    # get叶180NOAAID_方便拼接csv

    original_df = pd.read_csv("../创新点2_对比sr_先平均无标准差_未筛选/scaler_sr_fits提取_未筛选.csv")
    noaaid_values = original_df['noaaID'][::40].reset_index(drop=True)
    class_values = original_df['CLASS'][::40].reset_index(drop=True)

    print("++++++++++++++++++ 拿出40的一个拼起来与概率++++++++++++++++++++++++++++++++++")
    # Step 3: Concatenate NOAAID values with the averaged probabilities
    average_df.insert(0, 'noaaID', noaaid_values)  # Insert the NOAAID column at the beginning
    average_df.insert(0, 'Class', class_values)  # Insert the NOAAID column at the beginning

    # Step 4: Save the combined result to a new CSV file
    combineDir="average_probabilities_with_noaaid"
    if not os.path.exists(combineDir):
        os.makedirs(combineDir)
    average_df.to_csv(f"./{combineDir}/{datasetnumber}_{JianceType}_average_probabilities_with_noaaid.csv", index=False, encoding="utf-8-sig")
    print("Averaged probabilities with NOAAID values saved to 'average_probabilities_with_noaaid.csv'")

    print("++++++++++++++++++ 根据NOAAID进行匹配拼接++++++++++++++++++++++++++++++++++")

    # 读取两个CSV文件
    sr_completed_path = rf"G:\本科\SHARP_23_24\sr_completed.csv"
    average_probabilities_path = rf"./{combineDir}/{datasetnumber}_{JianceType}_average_probabilities_with_noaaid.csv"

    sr_data = pd.read_csv(sr_completed_path, encoding='ISO-8859-1')
    average_probabilities_data = pd.read_csv(average_probabilities_path)

    # 选择第一个CSV的`noaaID`和`MPlus`列
    sr_selected_data = sr_data[['noaaID', 'MPlus']]

    # 将第二个CSV文件的 `noaaID` 列设置为索引，以便按 `noaaID` 进行匹配
    average_probabilities_data.set_index('noaaID', inplace=True)

    # 创建一个空列表来存储匹配后的数据
    combined_data = []

    # 遍历第一个CSV文件的每一行
    for index, row in sr_selected_data.iterrows():
        noaaID = row['noaaID']
        MPlus = row['MPlus']

        # 获取第二个CSV文件中的 `zero` 和 `one` 值
        if noaaID in average_probabilities_data.index:
            zero = average_probabilities_data.loc[noaaID, 'zero']
            one = average_probabilities_data.loc[noaaID, 'one']
            Class = average_probabilities_data.loc[noaaID, 'Class']
        else:
            zero, one, Class = None, None, None  # 若未找到匹配，使用空值

        # 将 `noaaID`, `MPlus`, `zero`, `one` 添加为一行
        combined_data.append([noaaID, MPlus, zero, one, Class])

    # 将匹配结果转换为DataFrame
    combined_df = pd.DataFrame(combined_data, columns=['noaaID', 'MPlus', 'zero', 'one', 'Class'])
    # 删除 'zero' 字段为空的行
    combined_df.dropna(subset=['zero'], inplace=True)
    # 保存为新的CSV文件.
    Duiqidirectory = "对齐CSV"
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(Duiqidirectory):
        os.makedirs(Duiqidirectory)
    output_path = f"./{Duiqidirectory}/{datasetnumber}_对齐CSV概率与complete_{JianceType}.csv"
    combined_df.to_csv(output_path, index=False)
    # 重新读取保存的文件
    combined_df = pd.read_csv(output_path)

    # 根据 'Class' 字段添加 'y_true' 列
    combined_df['y_true'] = combined_df['Class'].apply(lambda x: 0 if x in ['N', 'C'] else 1)

    # 将包含 y_true 列的 DataFrame 再次保存到同一文件
    combined_df.to_csv(output_path, index=False)

    print(f"文件已更新，添加 y_true 列，并保存至 '{output_path}'")



    # os.remove("./average_probabilities_with_noaaid.csv")


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
def getOneDuiqiTss(Path,ProName,JinaceType):
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
            tssList.append(thisThresholdTss)

        return tssList
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
    y_true = df['y_true'].tolist()
    y_prob = np.array(df[f'{ProbName}'].tolist())

    BS, BSS = BS_BSS_score(y_true, y_prob)
    # 计算每个指标，确保每个指标为包含类别 [0] 和 [1] 的列表
    metrics = {
        "Accuracy": [metric.Accuracy()[0], metric.Accuracy()[1]],  # 准确率
        "Recall": [metric.Recall()[0], metric.Recall()[1]],        # 召回率
        "Precision": [metric.Precision()[0], metric.Precision()[1]],  # 精确率
        "TSS": [metric.TSS()[0], metric.TSS()[1]],                 # 真负率减去假正率
        "BSS": [ BS, BSS],                 # Brier技能得分
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
def getOnemodel_BestTssMetrics(ProbName,JianceType,datasetnumber):

    results = []  # 存储所有结果的列表
    # 使用for循环从0开始，增加至1（包含1）.
    step_size=0.05
    # 初始化起始值
    start = 0.05
    step_values = []
    for i in range(int(1 / step_size) + 1):  # +1 是为了确保1被包括进去
        step_value = start + i * step_size
        if step_value > 0.96:
            break  # 避免由于浮点精度问题超过1的情况
        step_values.append(step_value)
    for threshold in step_values:
        # 获取当前阈值下的所有指标
        metrics = getMetricsFromFile(rf"./对齐CSV/{datasetnumber}_对齐CSV概率与complete_{JianceType}.csv", threshold,ProbName)
        results.append([threshold, metrics])
    return results
    # 保存为CSV文件
    # output_path = f"{ProbName}_{JianceType}_BestTSS_metrics_results.csv"
    # df_results.to_csv(output_path, index=False)
    # print(f"结果已保存到 {output_path}")
def getBestTss_BSSMetrics(threshold,ProbName,JianceType):

    results = []  # 存储所有结果的列表
    for threshold in [threshold]:
        # 获取当前阈值下的所有指标
        metrics = getBestTss_BSSMetricsFromFile(fr"./对齐CSV概率与complete_{JianceType}.csv", threshold,ProbName)
        for metric_name, metric_value in metrics.items():
            # 将每个指标的类别 [0] 和 [1] 值添加到结果列表中
            results.append([threshold, metric_name, metric_value[0], metric_value[1]])
    # 创建包含指定结构的 DataFrame
    df_results = pd.DataFrame(results,
                              columns=["Threshold", "Metric Name/BS and BSS ", "Class 0 Value", "Class 1 Value"])
    print(df_results)
    # 保存为CSV文件
    output_path = f"{ProbName}_{JianceType}_BestTSS_metrics_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"结果已保存到 {output_path}")

def calculate_metrics_stats(tenmodelallMetricList,JianceType,ProName):
    from collections import defaultdict
    import numpy as np

    # 创建字典来存储指标数据
    results = defaultdict(lambda: defaultdict(list))

    # 遍历每个模型，并且为每个阈值收集数据
    for model_data in tenmodelallMetricList.values():
        # print(model_data) #拿到每个模型的阈值和指标二维列表
        for threshold, metrics in model_data: #阈值，{'Accuracy': [0.23809523809523808, 0.23809523809523808], 'Re}
            for metric, values in metrics.items(): #拿到每个阈值后面的集合分开 values =[0.23809523809523808, 0.23809523809523808]
                # Append values for this metric at this threshold
                results[threshold][metric].append(values)
    # print(results[0.05]){'Accuracy': [[0.23809523809523808, 0.23809523809523808], [0.23809523809523808, 0.238
    # 计算每个阈值和指标的均值和标准差
    stats = defaultdict(dict)
    max_tss_value = -float('inf')
    best_threshold = None
    for threshold, metrics in results.items():
        for metric, values in metrics.items():
            # Convert list of lists to a 2D numpy array
            values_array = np.array(values)
            # Calculate mean and std along the column (axis=0)
            mean = np.mean(values_array, axis=0)
            std = np.std(values_array, axis=0)
            # Store the results
            stats[threshold][metric] = {"mean": mean, "std": std}
            # Check if this metric is TSS and the second element is greater
            if metric == 'TSS' and mean[1] > max_tss_value:
                max_tss_value = mean[1]
                best_threshold = threshold
        # 保存最佳阈值下的所有指标数据到 CSV
    # 保存最佳阈值下的所有指标数据到 CSV
    # 保存最佳阈值下的所有指标数据到 CSV
    if best_threshold:
        # 准备数据，每个指标一行
        data_rows = []
        for metric, values in stats[best_threshold].items():
            mean_std_1 = f"{    truncate(values['mean'][0], 3)}_{    truncate(values['std'][0], 3)}"
            mean_std_2 = f"{    truncate(values['mean'][1], 3)}_{    truncate(values['std'][1], 3)}"
            data_rows.append([metric, mean_std_1, mean_std_2])

        # 创建 DataFrame
        df = pd.DataFrame(data_rows, columns=["Metric", "Class 1 Mean_Std", "Class 2 Mean_Std"])

        # 添加阈值信息作为单独一列
        df.insert(0, 'Threshold', f"{best_threshold}")

        # 写入 CSV 文件
        df.to_csv(f"{ProName}_{JianceType}_BestTSS_metrics_results.csv", index=True)


        # 创建包含指定结构的 DataFrame
        df_results = pd.DataFrame(data_rows, columns=["Metric", "负类", "正类"])
        # 保存为 Excel 文件
        output_excel_path = f"result_{ProName}_{JianceType}.xlsx"
        df_results.to_excel(output_excel_path, index=False, engine='openpyxl')  # 使用 openpyxl 保存为 .xlsx 格式
        print(f"结果已保存到 Excel: {output_excel_path}")
    return stats


def calculate_tss_statistics(tss_list,JianceType,ProName):
    """
    计算二维列表中每个位置的 TSS 均值和标准差，并保存结果到 CSV 文件。

    参数:
    tss_list (list of list of float): 二维列表，外层列表嵌套了在不同阈值下的 TSS 数据列表。
    """
    # 生成位置的列表
    step_value = len(tss_list[0])
    step_size = 0.05
    # 初始化起始值
    start = 0.05
    step_values = []
    for i in range(int(1 / step_size) + 1):  # +1 是为了确保1被包括进去
        step_value = start + i * step_size
        if step_value > 0.96:
            break  # 避免由于浮点精度问题超过1的情况
        step_values.append(step_value)
    thresholds = [f"{i}" for i in step_values]

    # 准备计算每个位置的 TSS 均值和标准差
    mean_list = []
    std_list = []

    # 将二维列表转置，使得每个列表包含所有位置相同索引的值
    transposed_tss = np.transpose(tss_list)

    # 遍历每个位置的数据，计算均值和标准差
    for position_data in transposed_tss:
        mean_list.append(truncate(np.mean(position_data), 3))
        std_list.append(truncate(np.std(position_data), 3))
    # 创建 DataFrame
    df = pd.DataFrame({
        "Threshold": thresholds,
        "TSS": mean_list,
        "TSSstd": std_list
    })

    # 保存到 CSV
    df.to_csv(f"threshold_tss_data_{ProName}_LLM_VIT_{JianceType}.csv", index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    print("=============================================================提取特征========================================================")
    # getTenFeature()
    print("=============================================================归一化特征========================================================")
    # globaldata()
    print("=============================================================测试数据集========================================================")
    # mainTestdataFile(["LLM_VIT"],"../创新点2_对比sr_先平均无标准差_未筛选/scaler_sr_fits提取_未筛选.csv","TSS")
    # mainTestdataFile(["LLM_VIT"],"../创新点2_对比sr_先平均无标准差_未筛选/scaler_sr_fits提取_未筛选.csv","BSS")
    print("=============================================================for循环开始直接逐个CSV对齐========================================================")
    allOneTssList=[]
    allMplusTssList=[]
    tenmodel_One_allMetricList={}
    tenmodel_Mplus_allMetricList={}

    for datasetnumber in range(10):
        duiqiOneandMPLus("TSS",datasetnumber)
        duiqiOneandMPLus("BSS",datasetnumber)
        print("=============================================================绘图One与Mplus概率阈值图十个计算========================================================")
        onetssList= getOneDuiqiTss(f"./对齐CSV/{datasetnumber}_对齐CSV概率与complete_TSS.csv","one","TSS")
        allOneTssList.append(onetssList)
        MPlustssList= getOneDuiqiTss(f"./对齐CSV/{datasetnumber}_对齐CSV概率与complete_TSS.csv","MPlus","TSS")
        allMplusTssList.append(MPlustssList)
        result_one=getOnemodel_BestTssMetrics("one","TSS",datasetnumber)
        tenmodel_One_allMetricList[datasetnumber]=result_one
        result_MPlus=getOnemodel_BestTssMetrics("MPlus","TSS",datasetnumber)
        tenmodel_Mplus_allMetricList[datasetnumber]=result_MPlus
    #接下来处理所有十个格式处理
    calculate_tss_statistics(allOneTssList,"TSS","one")
    calculate_tss_statistics(allMplusTssList,"TSS","MPlus")
    stats=calculate_metrics_stats(tenmodel_One_allMetricList,"TSS","one",)
    stats2=calculate_metrics_stats(tenmodel_Mplus_allMetricList,"TSS","MPlus",)

    allOneBssList=[]
    allMplusBssList=[]
    tenmodel_One_allMetricList_BSS={}
    tenmodel_Mplus_allMetricList_BSS={}
    for datasetnumber in range(10):
        # duiqiOneandMPLus("TSS",datasetnumber)
        # duiqiOneandMPLus("BSS",datasetnumber)
        print("=============================================================绘图One与Mplus概率阈值图十个计算========================================================")
        onetssList= getOneDuiqiTss(f"./对齐CSV/{datasetnumber}_对齐CSV概率与complete_BSS.csv","one","BSS")
        allOneBssList.append(onetssList)
        MPlustssList= getOneDuiqiTss(f"./对齐CSV/{datasetnumber}_对齐CSV概率与complete_BSS.csv","MPlus","BSS")
        allMplusBssList.append(MPlustssList)

        result_one=getOnemodel_BestTssMetrics("one","BSS",datasetnumber)
        tenmodel_One_allMetricList_BSS[datasetnumber]=result_one
        result_MPlus=getOnemodel_BestTssMetrics("MPlus","BSS",datasetnumber)
        tenmodel_Mplus_allMetricList_BSS[datasetnumber]=result_MPlus

    #接下来处理所有十个格式处理
    calculate_tss_statistics(allOneBssList,"BSS","one")
    calculate_tss_statistics(allMplusBssList,"BSS","MPlus")
    stats=calculate_metrics_stats(tenmodel_One_allMetricList_BSS,"BSS","one",)
    stats2=calculate_metrics_stats(tenmodel_Mplus_allMetricList_BSS,"BSS","MPlus",)




































    # print(stats[0.6000000000000001])#TSS,0.6088_0.0623,0.6088_0.0623

        # BSSmax_one_index= getOneDuiqiTss(f"./{datasetnumber}_对齐CSV概率与complete_BSS.csv","one","BSS")
        # BSSmax_MPlus_index= getOneDuiqiTss(f"./{datasetnumber}_对齐CSV概率与complete_BSS.csv","MPlus","BSS")
        # print("=============================================================绘图One与Mplus最优TSS指标计算========================================================")
        # getBestTssMetrics(max_one_index,"one","TSS")
        # getBestTssMetrics(max_MPlus_index,"MPlus","TSS")
        # getBestTss_BSSMetrics(BSSmax_one_index,"one","BSS")
        # getBestTss_BSSMetrics(BSSmax_MPlus_index,"MPlus","BSS")
