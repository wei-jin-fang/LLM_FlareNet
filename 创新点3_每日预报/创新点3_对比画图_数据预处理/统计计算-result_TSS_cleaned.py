import pandas as pd


def count_noaaID(file_path):
    """
    计算CSV文件中noaaID的总个数和去重后的个数

    参数:
        file_path (str): CSV文件路径

    返回:
        tuple: (总个数, 去重后的个数)
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 检查是否存在noaaID列
        if 'noaaID' not in df.columns:
            raise ValueError("CSV文件中没有找到 'noaaID' 列")

        # 计算总个数和去重后的个数
        total_count = len(df['noaaID'])
        unique_count = df['noaaID'].nunique()

        return total_count, unique_count
    except Exception as e:
        raise RuntimeError(f"处理文件时出错: {e}")


def count_t_rec(file_path):
    """
    计算CSV文件中T_REC列的总个数和去重后的个数

    参数:
        file_path (str): CSV文件路径

    返回:
        tuple: (总个数, 去重后的种类数)
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 检查是否存在T_REC列
        if 'T_REC' not in df.columns:
            raise ValueError("CSV文件中没有找到 'T_REC' 列")

        # 计算总个数和去重后的种类数
        total_count = len(df['T_REC'])
        unique_count = df['T_REC'].nunique()

        return total_count, unique_count
    except Exception as e:
        raise RuntimeError(f"处理文件时出错: {e}")

# 示例用法
file_path= r"../对比sr_单/result_forecast_TSS_cleaned.csv"
print("============================",file_path,"对齐之前","============================")
total, unique = count_noaaID(file_path)
print(f"noaaid总个数: {total}")
print(f"naaid去重后的个数: {unique}")
total, unique = count_t_rec(file_path)
print(f"T_REC天数总个数: {total}")
print(f"T_REC天数去重后的种类数: {unique}")
print("============================",file_path,"对齐之后","============================")
file_path= r"../对比sr_单/对齐CSV概率与complete_TSS.csv"
total, unique = count_noaaID(file_path)
print(f"noaaid总个数: {total}")
print(f"naaid去重后的个数: {unique}")
total, unique = count_t_rec(file_path)
print(f"T_REC天数总个数: {total}")
print(f"T_REC天数去重后的种类数: {unique}")