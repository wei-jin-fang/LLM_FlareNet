from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime, timedelta

def Globaldata_needPath(list_data):
    print("++++++++++++++++++++++++++++++++第四步归一化+++++++++++++++++++++++++++++++++++++++++++++")

    # 将数据加载到 DataFrame 中，并将所有数据转换为浮点数类型
    df = pd.DataFrame(list_data,
                      columns=["TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH",
                               "SAVNCPP",
                               "USFLUX", "AREA_ACR", "MEANPOT", "R_VALUE", "SHRGT45"])
    df = df.apply(pd.to_numeric)  # 转换为数值类型

    import os

    waitforStandfilepath = "E:\conda_code_tf\LLM\LLM_VIT\获取数据集\data\merged0_output.csv"
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

    # # 输出标准化的均值和方差
    # print("标准化的均值 (Mean):")
    # print(scaler.mean_)
    # print("\n标准化的方差 (Variance):")
    # print(scaler.var_)

    # 使用之前的scaler对新数据中的指定列进行标准化
    df_normalized = df.copy()
    new_data_normalized_data = scaler.transform(df_normalized)

    return new_data_normalized_data
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def Globaldata(list_data):
    print("++++++++++++++++++++++++++++++++第四步归一化+++++++++++++++++++++++++++++++++++++++++++++")

    # 将数据加载到 DataFrame 中，并将所有数据转换为浮点数类型
    df = pd.DataFrame(list_data,
                      columns=["TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH",
                               "SAVNCPP", "USFLUX", "AREA_ACR", "MEANPOT", "R_VALUE", "SHRGT45"])
    df = df.apply(pd.to_numeric)  # 转换为数值类型

    # 手动设置标准化参数（使用提供的均值和标准差）
    mean_values = np.array([
        1.19527086e+03, 2.68865850e+23, 2.45100336e+13, 1.50900269e+02,
        6.72536197e+12, 1.54981119e+22, 8.31207961e+02, 6.51523418e+03,
        3.13289555e+00, 2.62529642e+01
    ])
    scale_values = np.array([
        1.36034771e+03, 4.14845493e+23, 2.63605181e+13, 2.70139847e+02,
        1.00889221e+13, 1.73280368e+22, 8.09170152e+02, 4.20950529e+03,
        1.45805498e+00, 1.61363519e+01
    ])

    # 初始化 StandardScaler 并手动设置参数
    scaler = StandardScaler()
    scaler.mean_ = mean_values  # 设置均值
    scaler.scale_ = scale_values  # 设置标准差
    scaler.var_ = scale_values ** 2  # 方差是标准差的平方

    # 使用手动设置的 scaler 对新数据进行标准化
    df_normalized = df.copy()
    new_data_normalized_data = scaler.transform(df_normalized)

    return new_data_normalized_data
def compare_goes_class(class1, class2):
    """
    比较两个 GOES_Class 字段，返回较大的一个。
    比较规则：N < C < M < X，对于同一个字母，比较后面的数字部分，数字越大类别越大。
    """
    # 定义类别顺序
    order = {'A':-2,'B':-1,'N': 0, 'C': 1, 'M': 2, 'X': 3}

    # 获取字母和数字部分
    letter1, number1 = class1[0], float(class1[1:])
    letter2, number2 = class2[0], float(class2[1:])

    # 先按字母顺序比较
    if order[letter1] != order[letter2]:
        return class1 if order[letter1] > order[letter2] else class2
    else:
        # 字母相同时，比较数字部分
        return class1 if number1 > number2 else class2
def find_overall_max_goes_class(noaa_ars_dict):
    """
    在所有 NOAA_ARS 中找到整体最大类别的 GOES_Class 和对应的 NOAAID。
    :param noaa_ars_dict: 每个 NOAA_ARS 的最大 GOES_Class 和 NOAAID 字典
    :return: 整体最大 GOES_Class 和对应的 NOAAID
    """
    overall_max_goes_class = None
    overall_max_noaa_id = None

    for noaa_ars, (goes_class, noaa_id) in noaa_ars_dict.items():
        if overall_max_goes_class is None or compare_goes_class(goes_class, overall_max_goes_class) == goes_class:
            overall_max_goes_class = goes_class
            overall_max_noaa_id = noaa_id

    return overall_max_goes_class, overall_max_noaa_id
def calculate_begintime(insertnum):
    # 基础的结束时间是23:59
    end_time = datetime.strptime("23:59", "%H:%M")

    # 每个时间块的时长（比如每个数据块占12分钟）
    time_block_duration = timedelta(minutes=12)

    # 计算距离23:59的时间差
    remaining_blocks = 120 - insertnum  # 剩余需要下载的块数

    # 计算开始时间
    begintime = end_time - time_block_duration * remaining_blocks

    return begintime.strftime("%H:%M")


import sys
from datetime import datetime, timedelta

# 自定义类，用于同时将输出写入控制台和文件
class DualStream:
    def __init__(self, console_stream, file_stream):
        # 明确指定控制台流，例如直接指定为原始 sys.stdout
        self.console_stream = console_stream
        self.file_stream = file_stream

    def write(self, message):
        # 输出到控制台（原始的 sys.stdout）
        if self.console_stream:
            self.console_stream.write(message)

        # 输出到文件
        if self.file_stream:
            self.file_stream.write(message)

    def flush(self):
        # 确保文件流没有关闭
        if self.console_stream:
            self.console_stream.flush()
        if self.file_stream and not self.file_stream.closed:
            self.file_stream.flush()
if __name__ == '__main__':
    list_data = [[1.19527086e+03, 2.68865850e+23, 2.45100336e+13, 1.50900269e+02,
                  6.72536197e+12, 1.54981119e+22, 8.31207961e+02, 6.51523418e+03,
                  3.13289555e+00, 2.62529642e+01]]  # 与均值相同的数据
    result = Globaldata(list_data)
    print(result)