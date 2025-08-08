import pandas as pd

# 定义表头
headers = [
    "T_REC", "HARPNUM", "NOAA_NUM", "NOAA_ARS", "TOTUSJH", "TOTPOT",
    "TOTUSJZ", "ABSNJZH", "SAVNCPP", "USFLUX", "AREA_ACR", "MEANGPOT",
    "R_VALUE", "SHRGT45", "Nmbr", "dataorder"
]

# 读取数据
data = pd.read_csv('./data/sharp_data_ten_feature.csv', header=None, names=headers)

# 步骤1：提取日期
data['Date'] = data['T_REC'].str.split('_').str[0]

# 步骤2：按日期和Nmbr分组，统计每组行数
grouped = data.groupby(['Date', 'Nmbr']).size().reset_index(name='count')
print("Grouped data:\n", grouped)

# 步骤3：过滤组（假设每组120行，可调整条件）
valid_groups = grouped[grouped['count'] == 120][['Date', 'Nmbr']]
print("Valid groups:\n", valid_groups)

import numpy as np
from sklearn.preprocessing import StandardScaler

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


# 步骤4：合并数据
filtered_data = pd.merge(data, valid_groups, on=['Date', 'Nmbr'], how='inner')
print("Filtered data before normalization:\n", filtered_data)


# 步骤5：按日期和Nmbr分组，按dataorder排序，应用归一化并切片
def process_group(group):
    # 先按dataorder升序排序
    group = group.sort_values(by='dataorder')

    # 提取需要归一化的列
    normalize_columns = ["TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH",
                         "SAVNCPP", "USFLUX", "AREA_ACR", "MEANGPOT", "R_VALUE", "SHRGT45"]
    data_to_normalize = group[normalize_columns].values.tolist()

    # 应用归一化
    normalized_data = Globaldata(data_to_normalize)

    # 更新组中的归一化数据
    group[normalize_columns] = normalized_data

    # 每3条取1条
    return group.iloc[::3]


sorted_data = filtered_data.groupby(['Date', 'Nmbr']).apply(process_group).reset_index(drop=True)
print("Sorted data after normalization and slicing:\n", sorted_data)

# 删除临时列 'Date'
sorted_data = sorted_data.drop(columns=['Date'])

# 将结果保存到新CSV文件
sorted_data.to_csv('daily_testing.csv', index=False)

print("处理完成。数据已保存到 'daily_testing.csv'。")