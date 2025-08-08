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
df.to_csv('../SC_waitcompare.csv', index=False)