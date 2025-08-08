import os
import pandas as pd
print("++++++++++++++++++++++++++++++++第一步进行数据清洗+++++++++++++++++++++++++++++++++++++++++++++")
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

# 示例用法
folder_path = r'../../../Data2原始数据'
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


def check_and_add_columns(folder_path):
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


# 示例用法
folder_path = output_folder  # 替换为你的文件夹路径
check_and_add_columns(folder_path)

print("++++++++++++++++++++++++++++++++第三步合并指定csv+++++++++++++++++++++++++++++++++++++++++++++")
import pandas as pd

# 声明csv文件路径列表
numdataset = 0
csv_files = [rf'{output_folder}\{numdataset}Train.csv',
             rf'{output_folder}\{numdataset}Val.csv',
             rf'{output_folder}\{numdataset}Test.csv',
             rf'{output_folder}\{numdataset+1}Train.csv',
             rf'{output_folder}\{numdataset+1}Val.csv',
             rf'{output_folder}\{numdataset+1}Test.csv'

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

waitforStandfilepath=fr"./merged{numdataset}_output.csv"
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
new_csv_path = "../fits_files_info.csv"  # 替换为新CSV文件的路径
new_data = pd.read_csv(new_csv_path)

# 使用之前的scaler对新数据中的指定列进行标准化
new_data_normalized = new_data.copy()
new_data_normalized[normalize_columns] = scaler.transform(new_data[normalize_columns])

# 保存归一化后的数据到新的CSV文件
output_csv_path = "../scaler_fits_files_info.csv"  # 设置输出文件的路径
new_data_normalized.to_csv(output_csv_path, index=False)

# 确认消息
print(f"归一化后的数据已经保存到 '{output_csv_path}'")








