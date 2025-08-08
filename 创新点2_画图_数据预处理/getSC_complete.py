'''
我现在有一个csv，是scientific_reports_solar_flares_predictions
里面有如下字段serial	prediction_date	jsoc_date	noaaID	harpnum	c_class	MPlus	m5_class	solarmonitor_date
按照如下要求处理数据：
1、依次读取每一行，把prediction_date字段的日期格式为（例如2023/1/20  0:01:00）转化为年月日字符串例如20230120
2.依次读取每一行，把noaaID 字段，里面的数据部分提取出来
3.依次读取每一行，把MPlus 列的百分数，转化为小数，如果这一行为N/A 就丢弃
形成一个新的csv
'''
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('G:\本科\SHARP_23_24\scientific_reports_solar_flares_predictions.csv')

# Step 1: 转换 prediction_date 字段为指定格式
df['prediction_date'] = pd.to_datetime(df['prediction_date'], errors='coerce').dt.strftime('%Y%m%d')

# Step 2: 提取 noaaID 字段的数据部分（假设要提取的是数字部分）
df['noaaID'] = df['noaaID'].str.extract('(\d+)')
print(df[0:5])
# Step 3: 转换 MPlus 列中的百分数为小数，并丢弃 N/A 行
df = df[df['MPlus'] != 'N/A']  # 丢弃 N/A 行
df = df.dropna(subset=['MPlus'])  # 丢弃 MPlus 列为空的行
df['MPlus'] = df['MPlus'].str.rstrip('%').astype(float) / 100  # 转换百分数为小数

# 保存处理后的数据为新的 CSV 文件
df.to_csv('processed_SC.csv', index=False)

print("数据处理完成，已保存为 processed_solar_flares_predictions.csv")
# 读取上述处理完成的csv记作SC.csv
# 我目前还有一个文件夹，文件夹里面有NCMX四个文件夹，四个文件夹里面有若干数字编号文件夹
# 每个数字编号文件夹里有40张fits扩展名的文件，请你依次遍历每一个数组文件夹，
# fits扩展名的文件名字例如：hmi.sharp_cea_720s.9632.20230607_082400_TAI.magnetogram请你倒着拿到.20230607_ 点与下划线直接的日期，注意前面是举例，你只需要每一个数字文件夹里面的最后一个fits后缀的文件作为整个文件夹内的代表日期
# 根据日期与数字文件夹名字，与SC.csv的prediction_date和noaaID分别对比
# 如何两者一致，把SC的数据保存到一个新的csv，命名为SC_complete.csv
import os
import pandas as pd
import re

# 读取 SC.csv 文件
sc_df = pd.read_csv('processed_SC.csv')
sc_df['prediction_date'] = sc_df['prediction_date'].astype(str)
sc_df['noaaID'] = sc_df['noaaID'].astype(str)
# 创建一个空的 DataFrame 用于存储匹配的数据
matched_data = pd.DataFrame(columns=sc_df.columns)

# 定义 NCMX 主文件夹路径
base_path = r'G:\本科\SHARP_23_24\23_24SHARPImage_40'

# 遍历 NCMX 文件夹下的 N、C、M、X 子文件夹
for folder in ['N', 'C', 'M', 'X']:
    subfolder_path = os.path.join(base_path, folder)

    # 遍历数字编号文件夹
    for num_folder in os.listdir(subfolder_path):
        num_folder_path = os.path.join(subfolder_path, num_folder)

        # 检查路径是否为文件夹
        if os.path.isdir(num_folder_path):
            fits_files = [f for f in os.listdir(num_folder_path) if f.endswith('.fits')]

            # 选择最后一个 .fits 文件，最后一张的日期是预报日期
            if fits_files:
                latest_fits_file = sorted(fits_files)[-1]
                print(latest_fits_file)

                # 提取日期信息
                match = re.search(r'\.(\d{8})_', latest_fits_file)
                if match:
                    fits_date = match.group(1)  # 例如 "20230607"
                    print(fits_date)

                    # 将文件夹名称和日期作为字符串来进行匹配
                    formatted_date = fits_date  # 格式为 "20230607"
                    active_region = str(num_folder)  # 将数字文件夹名称作为字符串

                    # 比较 prediction_date 和 noaaID 是否匹配
                    condition = (sc_df['prediction_date'] == formatted_date) & \
                                (sc_df['noaaID'] == active_region)

                    # 获取匹配的行
                    matched_rows = sc_df[condition]
                    # 如果有匹配数据，则将其添加到结果 DataFrame
                    if not matched_rows.empty:
                        matched_data = pd.concat([matched_data, matched_rows])
exit()
# 保存匹配的数据到新的 CSV 文件
matched_data.to_csv('G:\本科\SHARP_23_24\sr_completed.csv', index=False)

