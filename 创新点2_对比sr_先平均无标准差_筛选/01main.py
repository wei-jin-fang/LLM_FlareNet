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

from run_main_Test import mainTestdataFile
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
def getTenShaixuanFeature():
    # 设置基础目录，即"23_24SHARPImage_40"文件夹的路径
    base_directory = r'G:\本科\SHARP_23_24\23_24SHARPImage_40'

    # 用于存储结果的列表
    fits_files_info = []

    # 需要的 header 键，包括NOAA_NUM检查
    required_keys = ['NOAA_NUM', 'TOTUSJH', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'USFLUX', 'AREA_ACR', 'MEANPOT',
                     'R_VALUE', 'SHRGT45']

    # 遍历 N、C、M、X 四个子文件夹
    for category in ['N', 'C', 'M', 'X']:
        category_folder = os.path.join(base_directory, category)

        # 检查类别文件夹是否存在
        if os.path.exists(category_folder):
            # 遍历数字命名的文件夹
            for numbered_folder in os.listdir(category_folder):
                numbered_folder_path = os.path.join(category_folder, numbered_folder)
                valid_folder = True  # 假设文件夹有效
                problematic_file = None  # 用于记录导致问题的文件名

                # 确保它是一个文件夹
                if os.path.isdir(numbered_folder_path):
                    # 列出数字文件夹中的所有 .fits 文件
                    fits_files = [f for f in os.listdir(numbered_folder_path) if f.endswith('.fits')]

                    # 临时列表来存储当前文件夹的信息
                    temp_info = []

                    # 检查当前文件夹内的所有文件
                    for fits_file in fits_files:
                        fits_file_path = os.path.join(numbered_folder_path, fits_file)

                        # 读取 .fits 文件的 header 信息
                        with fits.open(fits_file_path) as hdul:
                            header = hdul[-1].header  # 获取最后一个扩展的 header

                            # 只提取所需的键值对
                            header_info = {key: header.get(key, None) for key in required_keys}

                            # 检查NOAA_NUM是否为1

                            if str(header_info['NOAA_NUM']) != '1':
                                valid_folder = False
                                problematic_file = fits_file  # 记录导致问题的文件名
                                break  # 如果发现不为1，停止处理这个文件夹

                            # 将 header 信息添加到临时列表中
                            temp_info.append({
                                'CLASS': category,
                                'noaaID': numbered_folder,
                                'FITS文件': fits_file,
                                **header_info  # 将提取的 header 信息作为列
                            })

                    # 输出文件夹的处理结果
                    folder_status = "全是1" if valid_folder else f"不全是1，问题文件：{problematic_file}"
                    print(f"文件夹 {numbered_folder} 的NOAA_NUM {folder_status}")

                    # 如果文件夹验证通过，添加到最终列表
                    if valid_folder:
                        fits_files_info.extend(temp_info)

    # 将结果转换为 DataFrame 便于查看
    fits_files_df = pd.DataFrame(fits_files_info)

    output_csv_path = r'sr_fits提取_筛选.csv'
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
    folder_path = r'../../Data2原始数据'
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
    new_csv_path = "sr_fits提取_筛选.csv"  # 替换为新CSV文件的路径
    new_data = pd.read_csv(new_csv_path)

    # 使用之前的scaler对新数据中的指定列进行标准化
    new_data_normalized = new_data.copy()
    new_data_normalized[normalize_columns] = scaler.transform(new_data[normalize_columns])

    # 保存归一化后的数据到新的CSV文件
    output_csv_path = "./scaler_sr_fits提取_筛选.csv"  # 设置输出文件的路径
    new_data_normalized.to_csv(output_csv_path, index=False)

    # 确认消息
    print(f"归一化后的数据已经保存到 '{output_csv_path}'")

    os.remove(fr"./merged{numdataset}_output.csv")
    if os.path.exists("./clean之后数据"):
        shutil.rmtree("./clean之后数据")

def duiqiOneandMPLus(JianceType):
    #       接下来是概率预报阈值对比
    file_paths = glob.glob(fr"./{JianceType}CSV/{JianceType}_LLM_VIT*.csv")
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

    original_df = pd.read_csv("./scaler_sr_fits提取_筛选.csv")
    noaaid_values = original_df['noaaID'][::40].reset_index(drop=True)
    class_values = original_df['CLASS'][::40].reset_index(drop=True)

    print("++++++++++++++++++ 拿出40的一个拼起来与概率++++++++++++++++++++++++++++++++++")
    # Step 3: Concatenate NOAAID values with the averaged probabilities
    average_df.insert(0, 'noaaID', noaaid_values)  # Insert the NOAAID column at the beginning
    average_df.insert(0, 'Class', class_values)  # Insert the NOAAID column at the beginning

    # Step 4: Save the combined result to a new CSV file
    average_df.to_csv("./average_probabilities_with_noaaid.csv", index=False, encoding="utf-8-sig")
    print("Averaged probabilities with NOAAID values saved to 'average_probabilities_with_noaaid.csv'")

    print("++++++++++++++++++ 根据NOAAID进行匹配拼接++++++++++++++++++++++++++++++++++")

    # 读取两个CSV文件
    sr_completed_path = rf"G:\本科\SHARP_23_24\sr_completed.csv"
    average_probabilities_path = rf"./average_probabilities_with_noaaid.csv"

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
    # 保存为新的CSV文件
    output_path = f"对齐CSV概率与complete_{JianceType}.csv"
    combined_df.to_csv(output_path, index=False)
    # 重新读取保存的文件
    combined_df = pd.read_csv(output_path)

    # 根据 'Class' 字段添加 'y_true' 列
    combined_df['y_true'] = combined_df['Class'].apply(lambda x: 0 if x in ['N', 'C'] else 1)

    # 将包含 y_true 列的 DataFrame 再次保存到同一文件
    combined_df.to_csv(output_path, index=False)

    print(f"文件已更新，添加 y_true 列，并保存至 '{output_path}'")



    os.remove("./average_probabilities_with_noaaid.csv")


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
            tssList.append(truncate(thisThresholdTss, 3))
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
        plt.text(max_threshold, max_tss, f'({max_threshold:.2f}, {truncate(max_tss, 3):.3f})', ha='right', va='bottom')
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
    # print("=============================================================提取特征========================================================")
    # getTenShaixuanFeature()
    # print("=============================================================归一化特征========================================================")
    # globaldata()
    # print("=============================================================测试数据集========================================================")
    # mainTestdataFile(["LLM_VIT"],"./scaler_sr_fits提取_筛选.csv","TSS")
    # mainTestdataFile(["LLM_VIT"],"./scaler_sr_fits提取_筛选.csv","BSS")
    # print("=============================================================第一种先取概率One平均进行对齐========================================================")
    # duiqiOneandMPLus("TSS")
    # duiqiOneandMPLus("BSS")
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
