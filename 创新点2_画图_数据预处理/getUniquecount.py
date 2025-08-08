import pandas as pd

# 读取CSV文件
df = pd.read_csv('E:\conda_code_tf\LLM\LLM_VIT\创新点整理版本\创新点2_对比SR_先平均无标准差_未筛选\ccmc_fits提取_未筛选.csv')

# 提取第二列（假设第二列的列名为'column2'，可以根据实际情况调整）
column_data = df.iloc[:, 1]

# 去重并计算去重后的数量
unique_count = column_data.nunique()

print(f"去重后的数据个数: {unique_count}")
# 180
