import re
import pymysql
#
# db_config = {
#         'host': '129.226.126.98',  # 数据库地址
#         'user': 'sloarflare',  # 数据库用户名
#         'password': 'YMY5TJ3MXiaeFcAb',  # 数据库密码
#         'database': 'sloarflare',  # 数据库名
#         'charset': 'utf8mb4'  # 字符集
#     }

db_config = {
        'host': 'localhost',  # 数据库地址
        'user': 'root',  # 数据库用户名
        'password': 'wjf666',  # 数据库密码
        'database': 'solarflare',  # 数据库名
        'charset': 'utf8mb4'  # 字符集
    }

# 创建全局连接对象
connection = pymysql.connect(**db_config)
def db_update(sql):
    """
    执行数据库更新操作（插入、更新、删除），返回受影响的行数。
    查询使用 cursor.execute
    :param sql: 需要执行的SQL语句
    :return: 受影响的行数，如果出错则返回-1
    """
    # 不再在这里创建connection，而是使用全局的connection
    cursor = None
    try:
        # 创建游标
        cursor = connection.cursor()

        # 执行SQL语句
        rows_affected = cursor.execute(sql)

        # 提交事务
        connection.commit()

        logging.info(f"SQL执行成功，受影响行数：{rows_affected}")
        return rows_affected
    except pymysql.MySQLError as e:
        logging.error(f"SQL执行失败，错误信息：{e}")
        connection.rollback()  # 回滚事务
        return -1
    finally:
        # 只关闭游标，不关闭连接
        if cursor:
            cursor.close()
def get_data_noaaars_and_Nmbr_in_sql(T_REC_base):
    """
        根据给定的T_REC_base前缀，查询sharp_data_ten_feature表中的NOAA_ARS和Nmbr字段，
        并按NOAA_ARS分组返回结果。
        """
    cursor = None
    results = {}
    try:
        cursor = connection.cursor()
        query_sql = """
                SELECT NOAA_ARS, Nmbr
                FROM sharp_data_ten_feature
                WHERE T_REC LIKE %s
            """
        cursor.execute(query_sql, (f"{T_REC_base}%",))  # 因为之前就算补充的话拼接了下载当前的，所有通配符可以模糊查询
        rows = cursor.fetchall()
        for noaa_ars, nmbr in rows:
            if noaa_ars not in results:
                results[noaa_ars] = set()
            results[noaa_ars].add(nmbr)
        return results
    except pymysql.MySQLError as e:
        logging.error(f"查询失败，错误信息：{e}")
        return {}
    finally:
        if cursor:
            cursor.close()

def get_noaa_ids_for_today(today):
    date_format_ar = today.strftime("%Y-%m-%d")  # 格式化为 "2024-11-01"

    NOAAIDList = []
    cursor = None
    try:
        # 使用全局connection创建游标
        cursor = connection.cursor()
        select_sql = """
        SELECT Nmbr FROM ar_flare_prediction 
        WHERE AR_info_time = %s
        """
        cursor.execute(select_sql, (date_format_ar,))
        results = cursor.fetchall()
        NOAAIDList = [row[0] for row in results]  # 提取 Nmbr 存入 NOAAIDList
    except Exception as e:
        logging.error(f"查询发生错误: {e}")
    finally:
        # 只关闭游标，不关闭连接
        if cursor:
            cursor.close()
    return NOAAIDList

def insert_ar_info(ar_info_data):
    """
    插入活动区信息到数据库中。如果该记录已存在则不插入。
    :param ar_info_data: 包含AR活动区信息的列表
    :return: 受影响的行数，如果出错则返回-1
    """
    # 判读是否在左右60度之内
    isIn60degree=""
    if  int(ar_info_data[3][-2:]) <= 60:
        isIn60degree="T"
    else:
        isIn60degree="F"
    ar_info_insert_sql = f"""
        INSERT INTO ar_flare_prediction (id, AR_info_time, Nmbr, Location, Lo, Area, Z, LL, NN, Mag_Type, is60, is120)
        SELECT "{ar_info_data[0]}", "{ar_info_data[1]}", "{ar_info_data[2]}", "{ar_info_data[3]}",
        "{ar_info_data[4]}", "{ar_info_data[5]}", "{ar_info_data[6]}", "{ar_info_data[7]}", "{ar_info_data[8]}",
        "{ar_info_data[9]}", "{isIn60degree}", ""
        WHERE NOT EXISTS (
            SELECT 1 FROM ar_flare_prediction WHERE AR_info_time = "{ar_info_data[1]}" AND Nmbr = "{ar_info_data[2]}"
        )
    """
    return db_update(ar_info_insert_sql)

def AR_info_data_process(file_path, year, month, day):
    columns = ['id', 'AR_info_time', 'Nmbr', 'Location', 'Lo', 'Area', 'Z', 'LL', 'NN', 'Mag_Type']
    AR_info_list = []
    with open(file_path) as f:
        lines = f.readlines()
    is_use = False
    for line in lines:
        line = line.replace('\n', '')
        if not is_use:
            if "Nmbr" in line:
                is_use = True
                continue
        if is_use:
            if "IA." in line:
                break
            a = line.split(' ')
            AR_info = [x.strip() for x in a if x.strip() != '']
            AR_info_list.append(AR_info)
    download_time = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d").strftime("%Y-%m-%d 00:00:00")
    AR_info_list = [[str(uuid.uuid4()), download_time, '1' + item[0]] + item[1:] for item in AR_info_list]
    for AR_info_data in AR_info_list:
        ar_info_insert_row = insert_ar_info(AR_info_data)
        if ar_info_insert_row >= 0:
            logging.info(
                f"活动区信息{AR_info_data[1]}的{AR_info_data[2]}活动区保存成功，数据库受影响行数：{ar_info_insert_row}。")
        else:
            logging.error(
                f"活动区信息{AR_info_data[1]}的{AR_info_data[2]}活动区保存失败，数据库受影响行数：{ar_info_insert_row}")

def update_is120_field(date_format_ar, NOAAID, isin120):
    """
    更新ar_flare_prediction表中的is120字段。
    :param date_format_ar: 需要更新的日期，格式为 "20241101"
    :param NOAAID: 活动区的编号
    :param isin120: 插入的字符串值 "T" 或 "F"
    :return: 受影响的行数，如果出错则返回-1
    """
    # 构建更新SQL语句
    update_sql = f"""
        UPDATE ar_flare_prediction
        SET is120 = "{isin120}"
        WHERE AR_info_time = "{date_format_ar}" AND Nmbr = "{NOAAID}"
    """
    return db_update(update_sql)

def check_is60_is120(date_format_ar, NOAAID):
    """
    查询ar_flare_prediction表中指定日期和NOAAID的记录，检查is60是否为'T'且is120是否为'120'。
    :param date_format_ar: 查询的日期，格式为 格式为 "YYYY-MM-DD HH:MM:SS"（如 "2025-08-11 00:00:00"）
    :param NOAAID: 活动区的编号
    :return: 如果is60为'T'且is120为'120'，则返回True，否则返回False
    """
    cursor = None
    try:
        cursor = connection.cursor()
        query_sql = f"""
            SELECT is60, is120 FROM ar_flare_prediction
            WHERE AR_info_time = "{date_format_ar}" AND Nmbr = "{NOAAID}"
        """
        cursor.execute(query_sql)
        result = cursor.fetchone()
        if result:
            is60, is120 = result
            return is60 == 'T' and is120 == '120'
        else:
            return False
    except pymysql.MySQLError as e:
        logging.error(f"查询失败，错误信息：{e}")
        return False
    finally:
        if cursor:
            cursor.close()

def get_120_ten_feather_data_in_sql(T_REC_base, NOAAID):
    """
    根据T_REC_base和NOAAID查询sharp_data_ten_feature表，获取指定字段的记录。
    """
    cursor = None
    try:
        cursor = connection.cursor()
        query_sql = """
            SELECT TOTUSJH, TOTPOT, TOTUSJZ, ABSNJZH, SAVNCPP,
                   USFLUX, AREA_ACR, MEANPOT, R_VALUE, SHRGT45
            FROM sharp_data_ten_feature
            WHERE T_REC LIKE %s AND Nmbr = %s
            ORDER BY dataorder ASC
        """
        cursor.execute(query_sql, (f"{T_REC_base}%", NOAAID))
        rows = cursor.fetchall()
        return rows
    except pymysql.MySQLError as e:
        logging.error(f"查询失败，错误信息：{e}")
        return []
    finally:
        if cursor:
            cursor.close()

def fetch_noaa_ars_nmbr(T_REC_base):
    """
    根据给定的T_REC_base前缀，查询sharp_data_ten_feature表中的NOAA_ARS和Nmbr字段，
    并按NOAA_ARS分组返回结果。
    """
    cursor = None
    results = {}
    try:
        cursor = connection.cursor()
        query_sql = """
            SELECT NOAA_ARS, Nmbr
            FROM sharp_data_ten_feature
            WHERE T_REC LIKE %s
        """
        cursor.execute(query_sql, (f"{T_REC_base}%",))
        rows = cursor.fetchall()
        for noaa_ars, nmbr in rows:
            if noaa_ars not in results:
                results[noaa_ars] = set()
            results[noaa_ars].add(nmbr)
        return results
    except pymysql.MySQLError as e:
        logging.error(f"查询失败，错误信息：{e}")
        return {}
    finally:
        if cursor:
            cursor.close()

def insert_result_data_to_sql(data):
    """
    插入预测数据到result_forecast表。
    :param data: 要插入的数据列表
    """
    cursor = None
    try:
        cursor = connection.cursor()
        check_sql = """
            SELECT COUNT(*) 
            FROM result_forecast
            WHERE T_REC = %s AND _modelType = %s AND JianceType = %s AND NOAA_ARS = %s
              AND CV0_prob = %s AND CV1_prob = %s AND CV2_prob = %s AND CV3_prob = %s
              AND CV4_prob = %s AND CV5_prob = %s AND CV6_prob = %s AND CV7_prob = %s
              AND CV8_prob = %s AND CV9_prob = %s AND Nmbr = %s AND CV_average = %s
              AND Class = %s AND new_noaaid = %s AND predictday = %s
        """
        cursor.execute(check_sql, data)
        result = cursor.fetchone()
        if result[0] == 0:  # 如果没有找到重复数据
            insert_sql = """
                INSERT INTO result_forecast (
                    T_REC, _modelType, JianceType, NOAA_ARS, CV0_prob, CV1_prob, CV2_prob, CV3_prob, CV4_prob,
                    CV5_prob, CV6_prob, CV7_prob, CV8_prob, CV9_prob, Nmbr, CV_average, Class, new_noaaid, predictday
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_sql, data)
            connection.commit()
        print(f"Data {data[3]}-{data[1]}-{data[2]} inserted successfully")
    except pymysql.MySQLError as e:
        logging.error(f"Failed to insert data, error: {e}")
        connection.rollback()
    finally:
        if cursor:
            cursor.close()

def update_forecast_data(T_REC, _modelType, JianceType, NOAA_ARS, Class, new_noaaid):
    """
    更新预测数据到result_forecast表，根据T_REC, _modelType, JianceType, NOAA_ARS字段更新Class和new_noaaid。
    :param T_REC: 预测记录时间
    :param _modelType: 模型类型
    :param JianceType: 检测类型
    :param Class: 要更新的Class值
    :param new_noaaid: 要更新的new_noaaid值
    :param NOAA_ARS: 要用来定位更新记录的NOAA_ARS值
    """
    cursor = None
    try:
        cursor = connection.cursor()
        update_sql = """
            UPDATE result_forecast
            SET Class = %s, new_noaaid = %s
            WHERE T_REC = %s AND _modelType = %s AND JianceType = %s AND NOAA_ARS = %s
        """
        cursor.execute(update_sql, (Class, new_noaaid, T_REC, _modelType, JianceType, NOAA_ARS))
        connection.commit()
        print(f"Data updated successfully: T_REC={T_REC}, ModelType={_modelType}, JianceType={JianceType}, NOAA_ARS={NOAA_ARS}")
    except pymysql.MySQLError as e:
        logging.error(f"Failed to update data, error: {e}")
        connection.rollback()
    finally:
        if cursor:
            cursor.close()

def insert_event_data(EventdataList):
    """
    插入特定数据到 Eventdata 表，不使用参数化查询。
    """
    for event in EventdataList:
        date_format_event = event[1][4:12]
        eventID = f"{date_format_event}"  # 格式化EventID为日期
        event6 = str(event[6]).strip()  # S05W07(3883)
        Derived_Position = str(event6[:6]).strip()
        match = re.search(r'\((\s*\d+\s*)\)', event6)
        if match != None:
            NOAA_ID = "1" + match.group(1).strip()
        else:
            NOAA_ID = event6
        year, month, day = event[2].split()[0].split('/')
        formatted_date = year + month + day
        insert_sql = f"""
            INSERT INTO Eventdata (EventID, EName, Start, Stop, Peak, GOES_Class, Derived_Position, NOAA_ID, Start_day)
            VALUES ('{eventID}', '{event[1]}', '{event[2]}', '{event[3]}', '{event[4]}', '{event[5]}', '{Derived_Position}', '{NOAA_ID}', '{formatted_date}')
        """
        result = db_update(insert_sql)
        if result == -1:
            logging.error(f"Error inserting data for Event ID: {eventID}")
        else:
            logging.info(f"Successfully inserted data for Event ID: {eventID}")


# 在上面的代码风格基础上
# print("============================6.1遍历NOAA_ARS，去Event找到最大值以及对应代表ID=============================")
#                     for thisgroupid in NOAA_ARS:
#                         '''
#                         根据date_format_event变量和thisgroupid变量用于查询eventdata表数据的Start_day字段和NOAA_ID进行匹配查询，拿到每一行记录
#                         的GOES_Class字段和NOAAID字段，
#                         GOES_Class这个字段是一个字母与一个小数组合，例如M1.2，对于匹配到的所有记录首先先提取出来NOAAID是转化为int类型，纯数字的记录
#                         对于纯数字的记录（因为进行了匹配所以剩下的数字都一样）进行找到最大类别GOES_Class字段，对比规则：N小于C小于M小于X，对于同一个字母下比较后面的数字部分
#                         数字部分越大类别越大，然后把找到的GOES_Class字段和NOAAID字段进行保存
#                         '''
#                         pass
#                     '''
#                     整个for结束后，对于多个thisgroupid所对应的NOAA_ARS 找到这个最大类别，对比规则对比规则：N小于C小于M小于X，对于同一个字母下比较后面的数字部分
#                         数字部分越大类别越大，然后把找到的GOES_Class字段和NOAAID字段进行保存
#                     '''
#
# 完成上述代码的补充实现
def find_max_goes_class_for_noaa_ars(date_format_event, noaa_ars_list):
    """
    遍历 NOAA_ARS 列表，在 Eventdata 表中找到每个 NOAA_ARS 对应的最大 GOES_Class 和代表的 NOAAID。
    :param date_format_event: 用于查询的日期
    :param noaa_ars_list: NOAA_ARS 列表
    :return: 每个 NOAA_ARS 对应的最大 GOES_Class 和 NOAAID 的字典
    """
    max_goes_classes = {}
    cursor = None

    for thisgroupid in noaa_ars_list:
        '''
        确实有这几个
        13880 20241109
        13883 20241109
        13884 20241109
        13886 20241109
        '''
        try:
            # 使用全局 connection 创建游标
            cursor = connection.cursor()
            select_sql = """
            SELECT GOES_Class, NOAA_ID
            FROM Eventdata
            WHERE Start_day = %s AND NOAA_ID = %s
            """
            cursor.execute(select_sql, (date_format_event, thisgroupid))
            results = cursor.fetchall()
            max_goes_class = "N0"  # 默认值为 "N0"（无数据时）
            max_noaa_id = int(thisgroupid)
            # print(results)  noaa_ars_list里面每一个都找到查询结果
            for row in results:
                goes_class, noaa_id = row
                # 检查 NOAA_ID 是否为纯数字，且将其转换为 int 类型
                if noaa_id.isdigit():
                    # 比较当前 GOES_Class 是否比 max_goes_class 更大
                    if max_goes_class is None or compare_goes_class(goes_class, max_goes_class) == goes_class:
                        max_goes_class = goes_class
                        max_noaa_id = int(noaa_id)  # 转换为 int 类型

            # 保存最大 GOES_Class 和对应的 NOAAID
            if max_goes_class is not None:
                max_goes_classes[thisgroupid] = (max_goes_class, max_noaa_id)

        except pymysql.MySQLError as e:
            logging.error(f"查询 NOAA_ARS {thisgroupid} 的最大 GOES_Class 发生错误: {e}")
        finally:
            if cursor:
                cursor.close()
                cursor = None  # 重置 cursor 以便下次循环使用

    return max_goes_classes
'''
    下面是有关于下载10维特征的数据代码
'''
import os.path
import time
from datetime import datetime, timedelta
from decimal import Decimal

import urllib3

# from utils.pymysql_util import db_query, db_update
import requests
import logging
import uuid
from tqdm import tqdm
import shutil
# from utils.drive_letter import driver_letter
# from utils.file_util import are_file_sizes_within_range_of_median, get_file_size_median
from dateutil.relativedelta import relativedelta

import pymysql
# 数据库连接信息

DATA_SERIES = "hmi.sharp_cea_720s_nrt"
# USER_EMAIL = "pcyan0520@gmail.com"
Columns = ["TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP",
           "USFLUX", "AREA_ACR", "MEANPOT", "R_VALUE", "SHRGT45"]
def sharp_image_simple_download_para(download_day: datetime, NOAA_ARS,begin,noaa_ars_para,noaa_num_para):
    """
    :param download_datetime: 下载的图像的时间点
    """
    previous_day = download_day - timedelta(days=1)
    res = None
    resultstr = ",".join(Columns)
    try:
        DATA_SERIES = "hmi.sharp_cea_720s_nrt"
        # DATA_SERIES = "hmi.sharp_cea_720s"
        download_url = "http://jsoc.stanford.edu/cgi-bin/ajax/jsocextinfo"
        download_param = {
            # 2012.06.02_17:57_TAI-2012.06.03_17:57_TAI
            # "ds": f'{DATA_SERIES}[][{download_day.strftime("%Y.%m.%d")}/1d][? NOAA_ARS ~ "{NOAA_ARS}"  ?]'+ '{magnetogram}',
            "ds": f'{DATA_SERIES}[][{previous_day.strftime("%Y.%m.%d")}_{begin}_TAI-{download_day.strftime("%Y.%m.%d")}_23:48_TAI][? NOAA_ARS ~ "{NOAA_ARS}"  ?]'+ '{magnetogram}',
            # "ds": f'{DATA_SERIES}[][{previous_day.strftime("%Y.%m.%d")}_15:48_TAI-{download_day.strftime("%Y.%m.%d")}_23:48_TAI][? NOAA_ARS ~ "{NOAA_ARS}"  ?]'+ '{magnetogram}',
            "op": "rs_list",
            "key": "T_REC,HARPNUM,NOAA_NUM,NOAA_ARS,"+resultstr, #NOAA_NUM有几个活动区  NOAA_ARS（个数与NOAA_NUM对应） 13869？或者还有xxx
            "seg": "magnetogram",
            "link": "**NONE**",
            "R": "1",
            "userhandle": "20230831_00004_1681003066593",
            "dbhost": "hmidb2"
        }
        while True:
            logging.info(f"{download_day}，{NOAA_ARS}活动区的数据开始请求下载地址。")
            res = requests.get(download_url, params=download_param, timeout=90)
            print(res)
            if res.status_code == 200:
                logging.info(f"{download_day}，{NOAA_ARS}活动区的数据下载地址请求成功。")
                break

            else:
                logging.warning(
                    f"下载{download_day}，{NOAA_ARS}活动区的数据请求失败，失败码{res.status_code}，准备重新请求")
                time.sleep(5)
        download_datas = res.json()
        rcount = download_datas.get('count', -1000)
        if rcount == -1000:
            logging.warning(f"{download_day}的{NOAA_ARS}活动区磁图获取错误,无法下载，下载返回数据为{download_datas}。")
            return -999
        if rcount <= 0:
            logging.warning(f"{download_day}的{NOAA_ARS}活动区磁图数量为{rcount},无法下载，下载返回数据为{download_datas}。")
            return 3
        record_time_list = download_datas['keywords'][0]['values']
        record_time_list = [date_str.replace('.', '').replace(':', '') for date_str in record_time_list]
        HARPNUM_list = download_datas['keywords'][1]['values']
        noaa_num_list = download_datas['keywords'][2]['values']
        noaa_ars_list = download_datas['keywords'][3]['values']
        # 处理记录时间，将download_day拼接到previous_day前面
        for i in range(len(record_time_list)):
            record_time = record_time_list[i]
            if record_time.startswith(previous_day.strftime("%Y%m%d")):  # 判断是否是previous_day
                # 拼接download_day到previous_day之前
                record_time_list[i] = f"{download_day.strftime('%Y%m%d')}_{record_time[:]}"
            # 不区分所有的都赋值那天主要的，不管是不是补的，反正要保证补的和实际那天一样
            noaa_num_list[i] = noaa_num_para
            noaa_ars_list[i] = noaa_ars_para

        TOTUSJH_list = download_datas['keywords'][4]['values']
        TOTPOT_list = download_datas['keywords'][5]['values']
        # print(TOTPOT_list)3.946527e+22
        TOTUSJZ_list = download_datas['keywords'][6]['values']
        ABSNJZH_list = download_datas['keywords'][7]['values']
        SAVNCPP_list = download_datas['keywords'][8]['values']
        USFLUX_list = download_datas['keywords'][9]['values']
        AREA_ACR_list = download_datas['keywords'][10]['values']
        MEANPOT_list = download_datas['keywords'][11]['values']
        R_VALUE_list = download_datas['keywords'][12]['values']
        SHRGT45_list = download_datas['keywords'][13]['values']
        insertnum=len(SHRGT45_list)
        print(noaa_ars_list[0])
        try:
            with connection.cursor() as cursor:
                # 插入 SQL 语句
                insert_sql = """
                        INSERT INTO sharp_data_ten_feature (
                            T_REC, HARPNUM, NOAA_NUM, NOAA_ARS, TOTUSJH, TOTPOT, TOTUSJZ, ABSNJZH,
                            SAVNCPP, USFLUX, AREA_ACR, MEANPOT, R_VALUE, SHRGT45, Nmbr  ,dataorder
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """

                # 查询是否已存在相同的 T_REC 和 NOAA_ARS 记录
                select_sql = """
                        SELECT COUNT(*) FROM sharp_data_ten_feature
                        WHERE T_REC = %s AND NOAA_ARS = %s
                        """

                data_to_insert = []
                dataorder=1

                for record_time, har_num, noaa_num, noaa_ars, totusjh, totpot, totusjz, absnjzh, savncpp, usflux, area_acr, meanpot, r_value, shrgt45 in zip(
                        record_time_list, HARPNUM_list, noaa_num_list, noaa_ars_list,
                        TOTUSJH_list, TOTPOT_list, TOTUSJZ_list, ABSNJZH_list,
                        SAVNCPP_list, USFLUX_list, AREA_ACR_list, MEANPOT_list,
                        R_VALUE_list, SHRGT45_list):

                    cursor.execute(select_sql, (record_time, noaa_ars))
                    result = cursor.fetchone()

                    # if result[0] == 0:  # 如果不存在，则添加到插入数据列表中
                    # if True:  # 如果不存在，则添加到插入数据列表中
                    data_to_insert.append((
                            record_time, har_num, noaa_num, noaa_ars,
                            str(Decimal(totusjh)), str(Decimal(totpot)), str(Decimal(totusjz)), str(Decimal(absnjzh)),
                            str(Decimal(savncpp)), str(Decimal(usflux)), str(Decimal(area_acr)), str(Decimal(meanpot)),
                            str(Decimal(r_value)), str(Decimal(shrgt45)), NOAA_ARS, dataorder # 假设Nmbr的值暂时为空
                        ))
                    dataorder += 1


                # 批量插入数据
                if data_to_insert:
                    cursor.executemany(insert_sql, data_to_insert)
                    connection.commit()  # 提交事务
        finally:
            if cursor:
                cursor.close()

        return insertnum
    except requests.exceptions.Timeout as e:
        logging.error(f"发生错误，请求超时，{e}")
        return -888
    except TimeoutError as e:
        logging.error(f"发生错误，请求超时，TimeoutError，{e}")
        return -888
    except urllib3.exceptions.NewConnectionError as e:
        logging.error(f"发生错误，请求超时，urllib3.exceptions.NewConnectionError，{e}")
        return -888
    except urllib3.exceptions.MaxRetryError as e:
        logging.error(f"发生错误，请求超时，urllib3.exceptions.MaxRetryError，{e}")
        return -888
    except Exception as e:
        logging.error(f"发生错误，{e}，下载{download_day}的{NOAA_ARS}活动区发生错误", exc_info=True)
        return -999
    finally:
        if res is not None:
            res.close()

def sharp_image_simple_download(download_day: datetime, NOAA_ARS,noaa_ars_para,noaa_num_para):
    """
    :param download_datetime: 下载的图像的时间点
    """
    previous_day = download_day - timedelta(days=1)
    res = None
    resultstr = ",".join(Columns)
    try:
        DATA_SERIES = "hmi.sharp_cea_720s_nrt"
        # DATA_SERIES = "hmi.sharp_cea_720s"
        download_url = "http://jsoc.stanford.edu/cgi-bin/ajax/jsocextinfo"
        download_param = {
            # 2012.06.02_17:57_TAI-2012.06.03_17:57_TAI
            "ds": f'{DATA_SERIES}[][{download_day.strftime("%Y.%m.%d")}/1d][? NOAA_ARS ~ "{NOAA_ARS}"  ?]'+ '{magnetogram}',
            # "ds": f'{DATA_SERIES}[][{download_day.strftime("%Y.%m.%d")}_00:00_TAI-{download_day.strftime("%Y.%m.%d")}_23:48_TAI][? NOAA_ARS ~ "{NOAA_ARS}"  ?]'+ '{magnetogram}',
            "op": "rs_list",
            "key": "T_REC,HARPNUM,NOAA_NUM,NOAA_ARS,"+resultstr, #NOAA_NUM有几个活动区  NOAA_ARS（个数与NOAA_NUM对应） 13869？或者还有xxx
            "seg": "magnetogram",
            "link": "**NONE**",
            "R": "1",
            "userhandle": "20230831_00004_1681003066593",
            "dbhost": "hmidb2"
        }
        while True:
            logging.info(f"{download_day}，{NOAA_ARS}活动区的数据开始请求下载地址。")
            res = requests.get(download_url, params=download_param, timeout=90)
            print("响应结果",res)
            if res.status_code == 200:
                logging.info(f"{download_day}，{NOAA_ARS}活动区的数据下载地址请求成功。")
                break

            else:
                logging.warning(
                    f"下载{download_day}，{NOAA_ARS}活动区的数据请求失败，失败码{res.status_code}，准备重新请求")
                time.sleep(5)
        download_datas = res.json()
        rcount = download_datas.get('count', -1000)
        if rcount == -1000:
            logging.warning(f"{download_day}的{NOAA_ARS}活动区磁图获取错误,无法下载，下载返回数据为{download_datas}。")
            return -999
        if rcount <= 0:
            logging.warning(f"{download_day}的{NOAA_ARS}活动区磁图数量为{rcount},无法下载，下载返回数据为{download_datas}。")
            return 3
        record_time_list = download_datas['keywords'][0]['values']
        record_time_list = [date_str.replace('.', '').replace(':', '') for date_str in record_time_list]
        HARPNUM_list = download_datas['keywords'][1]['values']
        noaa_num_list = download_datas['keywords'][2]['values']
        noaa_ars_list = download_datas['keywords'][3]['values']

        # 处理记录时间，将download_day拼接到previous_day前面
        for i in range(len(record_time_list)):
            # 不区分所有的都赋值那天主要的，不管是不是补的，反正要保证补的和实际那天一样
            noaa_num_list[i] = noaa_num_para
            noaa_ars_list[i] = noaa_ars_para


        TOTUSJH_list = download_datas['keywords'][4]['values']
        TOTPOT_list = download_datas['keywords'][5]['values']
        # print(TOTPOT_list)3.946527e+22
        TOTUSJZ_list = download_datas['keywords'][6]['values']
        ABSNJZH_list = download_datas['keywords'][7]['values']
        SAVNCPP_list = download_datas['keywords'][8]['values']
        USFLUX_list = download_datas['keywords'][9]['values']
        AREA_ACR_list = download_datas['keywords'][10]['values']
        MEANPOT_list = download_datas['keywords'][11]['values']
        R_VALUE_list = download_datas['keywords'][12]['values']
        SHRGT45_list = download_datas['keywords'][13]['values']
        insertnum=len(SHRGT45_list)
        # print(noaa_ars_list[0])
        try:
            with connection.cursor() as cursor:
                # 插入 SQL 语句
                insert_sql = """
                        INSERT INTO sharp_data_ten_feature (
                            T_REC, HARPNUM, NOAA_NUM, NOAA_ARS, TOTUSJH, TOTPOT, TOTUSJZ, ABSNJZH,
                            SAVNCPP, USFLUX, AREA_ACR, MEANPOT, R_VALUE, SHRGT45, Nmbr,dataorder  
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """

                # 查询是否已存在相同的 T_REC 和 NOAA_ARS 记录
                select_sql = """
                        SELECT COUNT(*) FROM sharp_data_ten_feature
                        WHERE T_REC = %s AND NOAA_ARS = %s
                        """

                data_to_insert = []
                dataorder = 1
                for record_time, har_num, noaa_num, noaa_ars, totusjh, totpot, totusjz, absnjzh, savncpp, usflux, area_acr, meanpot, r_value, shrgt45 in zip(
                        record_time_list, HARPNUM_list, noaa_num_list, noaa_ars_list,
                        TOTUSJH_list, TOTPOT_list, TOTUSJZ_list, ABSNJZH_list,
                        SAVNCPP_list, USFLUX_list, AREA_ACR_list, MEANPOT_list,
                        R_VALUE_list, SHRGT45_list):

                    cursor.execute(select_sql, (record_time, noaa_ars))
                    result = cursor.fetchone()

                    # if result[0] == 0:  # 如果不存在，则添加到插入数据列表中
                    data_to_insert.append((
                            record_time, har_num, noaa_num, noaa_ars,
                            str(Decimal(totusjh)), str(Decimal(totpot)), str(Decimal(totusjz)), str(Decimal(absnjzh)),
                            str(Decimal(savncpp)), str(Decimal(usflux)), str(Decimal(area_acr)), str(Decimal(meanpot)),
                            str(Decimal(r_value)), str(Decimal(shrgt45)), NOAA_ARS,dataorder  # 假设Nmbr的值暂时为空
                        ))
                    dataorder+=1

                # 批量插入数据
                if data_to_insert:
                    cursor.executemany(insert_sql, data_to_insert)
                    connection.commit()  # 提交事务
        finally:
            pass
            if cursor:
                cursor.close()

        return insertnum
    except requests.exceptions.Timeout as e:
        logging.error(f"发生错误，请求超时，{e}")
        return -888
    except TimeoutError as e:
        logging.error(f"发生错误，请求超时，TimeoutError，{e}")
        return -888
    except urllib3.exceptions.NewConnectionError as e:
        logging.error(f"发生错误，请求超时，urllib3.exceptions.NewConnectionError，{e}")
        return -888
    except urllib3.exceptions.MaxRetryError as e:
        logging.error(f"发生错误，请求超时，urllib3.exceptions.MaxRetryError，{e}")
        return -888
    except Exception as e:
        logging.error(f"发生错误，{e}，下载{download_day}的{NOAA_ARS}活动区发生错误", exc_info=True)
        return -999
    finally:
        if res is not None:
            res.close()

def sharp_image_simple_getnumber(download_day: datetime, NOAA_ARS):
    """
    :param download_datetime: 下载的图像的时间点
    """
    previous_day = download_day - timedelta(days=1)
    res = None
    resultstr = ",".join(Columns)
    try:
        DATA_SERIES = "hmi.sharp_cea_720s_nrt"
        # DATA_SERIES = "hmi.sharp_cea_720s"
        download_url = "http://jsoc.stanford.edu/cgi-bin/ajax/jsocextinfo"
        download_param = {
            # 2012.06.02_17:57_TAI-2012.06.03_17:57_TAI
            # "ds": f'{DATA_SERIES}[][{download_day.strftime("%Y.%m.%d")}/1d][? NOAA_ARS ~ "{NOAA_ARS}"  ?]'+ '{magnetogram}',
            "ds": f'{DATA_SERIES}[][{download_day.strftime("%Y.%m.%d")}_00:00_TAI-{download_day.strftime("%Y.%m.%d")}_23:48_TAI][? NOAA_ARS ~ "{NOAA_ARS}"  ?]'+ '{magnetogram}',
            "op": "rs_list",
            "key": "T_REC,HARPNUM,NOAA_NUM,NOAA_ARS,"+resultstr, #NOAA_NUM有几个活动区  NOAA_ARS（个数与NOAA_NUM对应） 13869？或者还有xxx
            "seg": "magnetogram",
            "link": "**NONE**",
            "R": "1",
            "userhandle": "20230831_00004_1681003066593",
            "dbhost": "hmidb2"
        }
        while True:
            logging.info(f"{download_day}，{NOAA_ARS}活动区的数据开始请求下载地址。")
            res = requests.get(download_url, params=download_param, timeout=90)
            # print(res)
            if res.status_code == 200:
                logging.info(f"{download_day}，{NOAA_ARS}活动区的数据下载地址请求成功。")
                break

            else:
                logging.warning(
                    f"下载{download_day}，{NOAA_ARS}活动区的数据请求失败，失败码{res.status_code}，准备重新请求")
                time.sleep(5)
        download_datas = res.json()
        rcount = download_datas.get('count', -1000)
        if rcount == -1000:
            logging.warning(f"{download_day}的{NOAA_ARS}活动区磁图获取错误,无法下载，下载返回数据为{download_datas}。")
            return -999,0,0
        if rcount <= 0:
            logging.warning(f"{download_day}的{NOAA_ARS}活动区磁图数量为{rcount},无法下载，下载返回数据为{download_datas}。")
            return 3,0,0
        record_time_list = download_datas['keywords'][0]['values']
        record_time_list = [date_str.replace('.', '').replace(':', '') for date_str in record_time_list]
        HARPNUM_list = download_datas['keywords'][1]['values']
        noaa_num_list = download_datas['keywords'][2]['values']
        noaa_ars_list = download_datas['keywords'][3]['values']
        TOTUSJH_list = download_datas['keywords'][4]['values']
        TOTPOT_list = download_datas['keywords'][5]['values']
        # print(TOTPOT_list)3.946527e+22
        TOTUSJZ_list = download_datas['keywords'][6]['values']
        ABSNJZH_list = download_datas['keywords'][7]['values']
        SAVNCPP_list = download_datas['keywords'][8]['values']
        USFLUX_list = download_datas['keywords'][9]['values']
        AREA_ACR_list = download_datas['keywords'][10]['values']
        MEANPOT_list = download_datas['keywords'][11]['values']
        R_VALUE_list = download_datas['keywords'][12]['values']
        SHRGT45_list = download_datas['keywords'][13]['values']
        insertnum=len(SHRGT45_list)

        numbers_set = set()
        # 遍历 noaa_ars_list 中的每个字符串
        for item in noaa_ars_list:
            # 分割每个字符串，提取数字，并加入集合
            numbers_set.update(item.split(','))
        # 将集合中的数字按逗号拼接成一个新的字符串
        result_string = ','.join(sorted(numbers_set, key=int))

        return insertnum,result_string,len(numbers_set)
    except requests.exceptions.Timeout as e:
        logging.error(f"发生错误，请求超时，{e}")
        return -888,0,0
    except TimeoutError as e:
        logging.error(f"发生错误，请求超时，TimeoutError，{e}")
        return -888,0,0
    except urllib3.exceptions.NewConnectionError as e:
        logging.error(f"发生错误，请求超时，urllib3.exceptions.NewConnectionError，{e}")
        return -888,0,0
    except urllib3.exceptions.MaxRetryError as e:
        logging.error(f"发生错误，请求超时，urllib3.exceptions.MaxRetryError，{e}")
        return -888,0,0
    except Exception as e:
        logging.error(f"发生错误，{e}，下载{download_day}的{NOAA_ARS}活动区发生错误", exc_info=True)
        return -999,0,0
    finally:
        if res is not None:
            res.close()


if __name__ == '__main__':
    pass