import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# 设置Matplotlib后端以避免弃用警告
# 应在导入pyplot之前设置后端
matplotlib.use('TkAgg')  # 可以尝试使用 'Agg' 作为替代

# 读取CSV文件的路径
file_path = r"E:\conda_code_tf\LLM\LLM_VIT\创新点整理版本\创新点3_每日预报\对比ccmc_单\对齐CSV概率与complete_BSS.csv"
file_path = r"E:\conda_code_tf\LLM\LLM_VIT\创新点整理版本\创新点3_每日预报\对比sr_单\对齐CSV概率与complete_BSS.csv"

try:
    # 尝试使用UTF-8编码读取CSV文件
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    # 如果UTF-8编码失败，尝试使用GBK编码
    df = pd.read_csv(file_path, encoding='gbk')

# 定义'Class'字段的排序顺序
class_order = ['N', 'C', 'M', 'X']

# 将'Class'字段转换为有序的分类类型
df['Class'] = pd.Categorical(df['Class'], categories=class_order, ordered=True)

# 根据'Class'字段进行排序，并重置索引
df_sorted = df.sort_values('Class').reset_index(drop=True)

# 将'one'、'y_true'和'MPlus'字段转换为数值类型，无法转换的值设为NaN
df_sorted['one'] = pd.to_numeric(df_sorted['one'], errors='coerce')
df_sorted['y_true'] = pd.to_numeric(df_sorted['y_true'], errors='coerce')
df_sorted['MPlus'] = pd.to_numeric(df_sorted['MPlus'], errors='coerce')

# 检查'one'、'y_true'或'MPlus'列中是否存在缺失值
if df_sorted[['one', 'y_true', 'MPlus']].isnull().any().any():
    print("Warning: Missing or non-numeric values detected in 'one', 'y_true', or 'MPlus' columns.")
    # 删除包含缺失值的行
    df_sorted = df_sorted.dropna(subset=['one', 'y_true', 'MPlus'])

# 计算'one'与'y_true'的绝对值差，并存储在新列'abs_diff_one_y_true'中
df_sorted['abs_diff_one_y_true'] = (df_sorted['one'] - df_sorted['y_true']).abs()

# 计算'y_true'与'MPlus'的绝对值差，并存储在新列'abs_diff_y_true_MPlus'中
df_sorted['abs_diff_y_true_MPlus'] = (df_sorted['y_true'] - df_sorted['MPlus']).abs()

# 设置Matplotlib的中文字体以避免字体缺失警告
# 请确保系统中已安装'SimHei'字体，或根据需要更换为其他支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 定义不同'Class'类别对应的颜色
color_mapping = {'N': 'red', 'C': 'green', 'M': 'orange', 'X': 'purple'}

# 创建一个包含两个子图的图形，设置图形大小
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 6))  # 1行2列

# ------------------ 第一个子图：|one - y_true|的散点图 ------------------
ax1 = axes[0]  # 第一个子图

# 遍历每个类别，绘制对应的散点
for cls in class_order:
    subset = df_sorted[df_sorted['Class'] == cls]
    ax1.scatter(
        subset.index,
        subset['abs_diff_one_y_true'],
        color=color_mapping.get(cls, 'b'),
        alpha=0.6,
        label=cls
    )

# 设置第一个子图的标题和轴标签
ax1.set_title('Absolute Difference (|one - y_true|) Scatter Plot')
ax1.set_xlabel('Row Number')
ax1.set_ylabel('|one - y_true|')

# 添加图例，标题为'Class'
ax1.legend(title='Class')

# 添加网格以增强可读性
ax1.grid(True)

# ------------------ 第二个子图：|y_true - MPlus|的散点图 ------------------
ax2 = axes[1]  # 第二个子图

# 遍历每个类别，绘制对应的散点
for cls in class_order:
    subset = df_sorted[df_sorted['Class'] == cls]
    ax2.scatter(
        subset.index,
        subset['abs_diff_y_true_MPlus'],
        color=color_mapping.get(cls, 'b'),
        alpha=0.6,
        label=cls
    )

# 设置第二个子图的标题和轴标签
ax2.set_title('Absolute Difference (|y_true - MPlus|) Scatter Plot')
ax2.set_xlabel('Row Number')
ax2.set_ylabel('|y_true - MPlus|')

# 添加图例，标题为'Class'
ax2.legend(title='Class')

# 添加网格以增强可读性
ax2.grid(True)

# 调整布局以防止标签被截断
plt.tight_layout()

# 显示图表
plt.show()

# 可选：将图表保存为PNG文件
fig.savefig('combined_scatter_plots_SR.png', dpi=300)
