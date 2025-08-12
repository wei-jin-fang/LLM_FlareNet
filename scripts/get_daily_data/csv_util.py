import csv
from typing import List, Tuple, Dict, Set

import pandas as pd
import os
import pandas as pd
import os
from typing import List, Tuple

def insert_result_data_to_csv(data: Tuple, csv_file_path: str = "./result_forecast.csv") -> bool:
    """
    插入预测数据到result_forecast.csv文件，插入前检查是否已存在相同记录。
    :param data: 要插入的数据元组，包含T_REC, _modelType, JianceType, NOAA_ARS, CV0_prob, CV1_prob, CV2_prob,
                 CV3_prob, CV4_prob, CV5_prob, CV6_prob, CV7_prob, CV8_prob, CV9_prob, Nmbr, CV_average, Class,
                 new_noaaid, predictday
    :param csv_file_path: CSV文件的路径，默认为 "./result_forecast.csv"
    :return: 成功返回True，失败返回False
    """
    try:
        # CSV文件的表头
        headers = ['T_REC', '_modelType', 'JianceType', 'NOAA_ARS', 'CV0_prob', 'CV1_prob', 'CV2_prob', 'CV3_prob',
                   'CV4_prob', 'CV5_prob', 'CV6_prob', 'CV7_prob', 'CV8_prob', 'CV9_prob', 'Nmbr', 'CV_average',
                   'Class', 'new_noaaid', 'predictday']

        # 确保数据为字符串并去除前后空格
        data = tuple(str(field).strip() for field in data)

        # 文件是否存在
        file_exists = os.path.exists(csv_file_path)

        # 读取现有CSV文件
        if file_exists:
            df = pd.read_csv(csv_file_path, encoding='utf-8', dtype=str)
        else:
            df = pd.DataFrame(columns=headers)

        # 检查必要的列是否存在
        if not all(col in df.columns for col in headers):
            print(f"CSV文件中缺少必要的列: {headers}")
            return False

        # 清理CSV中的数据：去除前后空格
        for col in headers:
            df[col] = df[col].str.strip()

        # 检查是否已存在相同记录
        condition = (
            (df['T_REC'] == data[0]) &
            (df['_modelType'] == data[1]) &
            (df['JianceType'] == data[2]) &
            (df['NOAA_ARS'] == data[3]) &
            (df['CV0_prob'] == data[4]) &
            (df['CV1_prob'] == data[5]) &
            (df['CV2_prob'] == data[6]) &
            (df['CV3_prob'] == data[7]) &
            (df['CV4_prob'] == data[8]) &
            (df['CV5_prob'] == data[9]) &
            (df['CV6_prob'] == data[10]) &
            (df['CV7_prob'] == data[11]) &
            (df['CV8_prob'] == data[12]) &
            (df['CV9_prob'] == data[13]) &
            (df['Nmbr'] == data[14]) &
            (df['CV_average'] == data[15]) &
            (df['Class'] == data[16]) &
            (df['new_noaaid'] == data[17]) &
            (df['predictday'] == data[18])
        )

        if not df[condition].empty:
            print(f"记录已存在，跳过: NOAA_ARS={data[3]}, _modelType={data[1]}, JianceType={data[2]}")
            return True

        # 追加新记录
        new_row = pd.DataFrame([data], columns=headers)
        df = pd.concat([df, new_row], ignore_index=True)

        # 写回CSV文件
        df.to_csv(csv_file_path, index=False, encoding='utf-8')

        print(f"Data {data[3]}-{data[1]}-{data[2]} inserted successfully")
        return True

    except Exception as e:
        print(f"写入CSV文件时出错: {e}")
        return False

def get_120_ten_feature_data_in_csv(T_REC_base: str, NOAAID: str, csv_file_path: str = "./sharp_data_ten_feature.csv") -> List[Tuple]:
    """
    根据T_REC_base和NOAAID查询sharp_data_ten_feature.csv，获取指定字段的记录。
    :param T_REC_base: 查询的日期前缀，格式为 "YYYY-MM-DD" 或更精确（如 "2025-08-11 00:00:00"），支持模糊匹配
    :param NOAAID: 活动区的编号（字符串）
    :param csv_file_path: CSV文件的路径，默认为 "./sharp_data_ten_feature.csv"
    :return: 包含查询结果的元组列表，每个元组包含TOTUSJH, TOTPOT, TOTUSJZ, ABSNJZH, SAVNCPP, USFLUX, AREA_ACR, MEANPOT, R_VALUE, SHRGT45
             如果查询失败或无记录，返回空列表
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(csv_file_path):
            print(f"CSV文件 {csv_file_path} 不存在")
            return []

        # 读取CSV文件，强制所有列为字符串
        df = pd.read_csv(csv_file_path, encoding='utf-8', dtype=str)

        # 检查必要的列是否存在
        required_columns = ['T_REC', 'Nmbr', 'TOTUSJH', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP',
                           'USFLUX', 'AREA_ACR', 'MEANPOT', 'R_VALUE', 'SHRGT45', 'dataorder']
        if not all(col in df.columns for col in required_columns):
            print(f"CSV文件中缺少必要的列: {required_columns}")
            return []

        # 清理输入数据：去除前后空格，转换为字符串
        T_REC_base = str(T_REC_base).strip()
        NOAAID = str(NOAAID).strip()

        # 清理CSV中的数据：去除前后空格
        df['T_REC'] = df['T_REC'].str.strip()
        df['Nmbr'] = df['Nmbr'].str.strip()

        # 模糊匹配 T_REC（模拟 SQL 的 LIKE）
        condition = (df['T_REC'].str.startswith(T_REC_base)) & (df['Nmbr'] == NOAAID)
        if not df[condition].empty:
            # 使用 .loc 直接修改原始 DataFrame 的子集
            df.loc[condition, 'dataorder'] = pd.to_numeric(df[condition]['dataorder'], errors='coerce')

        # 获取匹配的行
        matching_rows = df[condition]

        if matching_rows.empty:
            print(f"未找到匹配的记录: T_REC_base='{T_REC_base}', Nmbr='{NOAAID}'")
            return []

        # 按 dataorder 升序排序
        matching_rows = matching_rows.sort_values(by='dataorder', ascending=True)

        # 提取所需字段
        result = matching_rows[['TOTUSJH', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP',
                               'USFLUX', 'AREA_ACR', 'MEANPOT', 'R_VALUE', 'SHRGT45']].to_records(index=False)

        # 将结果转换为元组列表
        return [tuple(row) for row in result]

    except Exception as e:
        print(f"查询CSV文件时出错: {e}")
        return []


def check_data_in_csv(date_format_ar: str, NOAAID: str, csv_file_path: str = "./ar_flare_prediction.csv") -> bool:
    """
    查询ar_flare_prediction.csv中指定日期和NOAAID的记录，检查is60是否为'T'且is120是否为'120'。
    :param date_format_ar: 查询的日期，格式为 "YYYY-MM-DD HH:MM:SS"（如 "2025-08-11 00:00:00"）
    :param NOAAID: 活动区的编号（字符串）
    :param csv_file_path: CSV文件的路径，默认为 "./ar_flare_prediction.csv"
    :return: 如果is60为'T'且is120为'120'，则返回True，否则返回False
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(csv_file_path):
            print(f"CSV文件 {csv_file_path} 不存在")
            return False

        # 读取CSV文件，强制所有列为字符串
        df = pd.read_csv(csv_file_path, encoding='utf-8', dtype=str)

        # 检查必要的列是否存在
        required_columns = ['AR_info_time', 'Nmbr', 'is60', 'is120']
        if not all(col in df.columns for col in required_columns):
            print(f"CSV文件中缺少必要的列: {required_columns}")
            return False

        # 清理输入数据：去除前后空格，转换为字符串
        date_format_ar = str(date_format_ar).strip()
        NOAAID = str(NOAAID).strip()

        # 清理CSV中的数据：去除前后空格
        df['AR_info_time'] = df['AR_info_time'].str.strip()
        df['Nmbr'] = df['Nmbr'].str.strip()
        df['is60'] = df['is60'].str.strip()
        df['is120'] = df['is120'].str.strip()

        # 找到匹配的行
        condition = (df['AR_info_time'] == date_format_ar) & (df['Nmbr'] == NOAAID)
        matching_rows = df[condition]

        if matching_rows.empty:
            print(f"未找到匹配的记录: AR_info_time='{date_format_ar}', Nmbr='{NOAAID}'")
            return False

        # 检查is60和is120
        is60, is120 = matching_rows.iloc[0]['is60'], matching_rows.iloc[0]['is120']
        return is60 == 'T' and is120 == '120'

    except Exception as e:
        print(f"查询CSV文件时出错: {e}")
        return False


def get_data_noaaars_and_Nmbr_in_csv(T_REC_base: str, csv_file_path: str = "./sharp_data_ten_feature.csv") -> Dict[str, Set[str]]:
    """
    根据给定的T_REC_base前缀，查询sharp_data_ten_feature.csv中的NOAA_ARS和Nmbr字段，
    并按NOAA_ARS分组返回结果。
    :param T_REC_base: 查询的日期前缀，格式为 "YYYY-MM-DD" 或更精确（如 "2025-08-11 00:00:00"），支持模糊匹配
    :param csv_file_path: CSV文件的路径，默认为 "./sharp_data_ten_feature.csv"
    :return: 字典，键为NOAA_ARS，值为对应的Nmbr集合（set）；查询失败或无记录返回空字典
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(csv_file_path):
            print(f"CSV文件 {csv_file_path} 不存在")
            return {}

        # 读取CSV文件，强制所有列为字符串
        df = pd.read_csv(csv_file_path, encoding='utf-8', dtype=str)

        # 检查必要的列是否存在
        required_columns = ['T_REC', 'NOAA_ARS', 'Nmbr']
        if not all(col in df.columns for col in required_columns):
            print(f"CSV文件中缺少必要的列: {required_columns}")
            return {}

        # 清理输入数据：去除前后空格，转换为字符串
        T_REC_base = str(T_REC_base).strip()

        # 清理CSV中的数据：去除前后空格
        df['T_REC'] = df['T_REC'].str.strip()
        df['NOAA_ARS'] = df['NOAA_ARS'].str.strip()
        df['Nmbr'] = df['Nmbr'].str.strip()

        # 模糊匹配 T_REC（模拟 SQL 的 LIKE）
        condition = df['T_REC'].str.startswith(T_REC_base)
        matching_rows = df[condition][['NOAA_ARS', 'Nmbr']]

        if matching_rows.empty:
            print(f"未找到匹配的记录: T_REC_base='{T_REC_base}'")
            return {}

        # 按 NOAA_ARS 分组，收集 Nmbr 集合
        results = {}
        for _, row in matching_rows.iterrows():
            noaa_ars = row['NOAA_ARS']
            nmbr = row['Nmbr']
            if noaa_ars not in results:
                results[noaa_ars] = set()
            results[noaa_ars].add(nmbr)

        return results

    except Exception as e:
        print(f"查询CSV文件时出错: {e}")
        return {}

def update_is120_in_csv(date_format_ar: str, NOAAID: str, isin120: str, csv_file_path: str = "./ar_flare_prediction.csv") -> int:
    """
    更新ar_flare_prediction.csv文件中的is120字段。
    :param date_format_ar: 需要更新的日期，格式为 "YYYY-MM-DD HH:MM:SS"（如 "2025-08-11 00:00:00"）
    :param NOAAID: 活动区的编号（字符串）
    :param isin120: 插入的字符串值 "T" 或 "F"
    :param csv_file_path: CSV文件的路径，默认为 "./ar_flare_prediction.csv"
    :return: 受影响的行数，如果出错则返回-1
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(csv_file_path):
            print(f"CSV文件 {csv_file_path} 不存在")
            return -1

        # 读取CSV文件
        df = pd.read_csv(csv_file_path, encoding='utf-8', dtype=str)  # 强制所有列为字符串

        # 检查必要的列是否存在
        required_columns = ['AR_info_time', 'Nmbr', 'is120']
        if not all(col in df.columns for col in required_columns):
            print(f"CSV文件中缺少必要的列: {required_columns}")
            return -1

        # 清理输入数据：去除前后空格，转换为字符串
        date_format_ar = str(date_format_ar).strip()
        NOAAID = str(NOAAID).strip()

        # 清理CSV中的数据：去除前后空格
        df['AR_info_time'] = df['AR_info_time'].str.strip()
        df['Nmbr'] = df['Nmbr'].str.strip()

        # 找到匹配的行，更新is120列
        condition = (df['AR_info_time'] == date_format_ar) & (df['Nmbr'] == NOAAID)
        affected_rows = len(df[condition])

        if affected_rows == 0:
            print(f"未找到匹配的记录: AR_info_time='{date_format_ar}', Nmbr='{NOAAID}'")
            # 打印CSV中的前几行以便调试
            print("CSV文件中的前几行数据：")
            print(df[['AR_info_time', 'Nmbr']].head().to_string())
            return 0

        # 更新is120字段
        df.loc[condition, 'is120'] = isin120

        # 将更新后的内容写回CSV文件
        df.to_csv(csv_file_path, index=False, encoding='utf-8')

        return affected_rows

    except Exception as e:
        print(f"更新CSV文件时出错: {e}")
        return -1

def append_sharp_data_to_csv(data_to_insert: List[Tuple], csv_file_path: str = "./sharp_data_ten_feature.csv") -> bool:
    """
    将SHARP数据追加写入CSV文件，写入前检查是否已存在相同的T_REC和NOAA_ARS记录。
    :param data_to_insert: 包含SHARP数据的列表，每个元素为一个元组，包含T_REC, HARPNUM, NOAA_NUM, NOAA_ARS, TOTUSJH, TOTPOT, TOTUSJZ, ABSNJZH, SAVNCPP, USFLUX, AREA_ACR, MEANPOT, R_VALUE, SHRGT45, Nmbr, dataorder
    :param csv_file_path: CSV文件的路径，默认为 "./sharp_data_ten_feature.csv"
    :return: 成功返回True，失败返回False
    """
    if data_to_insert==None:
        return False
    try:
        # CSV文件的表头
        headers = ['T_REC', 'HARPNUM', 'NOAA_NUM', 'NOAA_ARS', 'TOTUSJH', 'TOTPOT', 'TOTUSJZ',
                   'ABSNJZH', 'SAVNCPP', 'USFLUX', 'AREA_ACR', 'MEANPOT', 'R_VALUE',
                   'SHRGT45', 'Nmbr', 'dataorder']

        # 文件是否存在
        file_exists = os.path.exists(csv_file_path)

        # 读取现有CSV文件中的记录，检查重复
        existing_records = set()
        if file_exists:
            with open(csv_file_path, mode='r', encoding='utf-8', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # 使用 T_REC 和 NOAA_ARS 作为唯一标识
                    existing_records.add((row['T_REC'], row['NOAA_ARS']))

        # 以追加模式打开CSV文件
        with open(csv_file_path, mode='a', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)

            # 如果文件不存在，写入表头
            if not file_exists:
                writer.writerow(headers)

            # 遍历data_to_insert，写入未存在的记录
            for record in data_to_insert:
                # 检查记录是否已存在
                if (record[0], record[3]) not in existing_records:
                    writer.writerow(record)
                    # 更新existing_records，避免重复检查
                    existing_records.add((record[0], record[3]))

        return True

    except Exception as e:
        print(f"写入CSV文件时出错: {e}")
        return False
def append_ar_flare_prediction_to_csv(AR_info_list: List[List[str]], csv_file_path: str = "./ar_flare_prediction.csv") -> bool:
    """
    将AR活动区信息追加写入CSV文件，写入前检查记录是否已存在。
    :param AR_info_list: 包含AR活动区信息的列表，每个元素为一个子列表，包含id, AR_info_time, Nmbr, Location, Lo, Area, Z, LL, NN, Mag_Type
    :param csv_file_path: CSV文件的路径，默认为 "./ar_flare_prediction.csv"
    :return: 成功返回True，失败返回False
    """
    if AR_info_list==None:
        return False
    try:
        # 判断是否在左右60度之内
        def is_in_60_degree(location: str) -> str:
            return "T" if int(location[-2:]) <= 60 else "F"

        # CSV文件的表头
        headers = ['id', 'AR_info_time', 'Nmbr', 'Location', 'Lo', 'Area', 'Z', 'LL', 'NN', 'Mag_Type', 'is60', 'is120']

        # 文件是否存在
        file_exists = os.path.exists(csv_file_path)

        # 读取现有CSV文件中的记录，检查重复
        existing_records = set()
        if file_exists:
            with open(csv_file_path, mode='r', encoding='utf-8', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # 使用 AR_info_time 和 Nmbr 作为唯一标识
                    existing_records.add((row['AR_info_time'], row['Nmbr']))

        # 以追加模式打开CSV文件
        with open(csv_file_path, mode='a', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)

            # 如果文件不存在，写入表头
            if not file_exists:
                writer.writerow(headers)

            # 遍历AR_info_list，写入未存在的记录
            for ar_info_data in AR_info_list:
                # 检查记录是否已存在
                if (ar_info_data[1], ar_info_data[2]) not in existing_records:
                    # 计算is60
                    is60 = is_in_60_degree(ar_info_data[3])
                    # 构造写入的行，is120默认空字符串
                    row = list(ar_info_data) + [is60, '']
                    writer.writerow(row)
                    # 更新existing_records，避免重复检查
                    existing_records.add((ar_info_data[1], ar_info_data[2]))

        return True

    except Exception as e:
        print(f"写入CSV文件时出错: {e}")
        return False