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
from dataprocess import calculate_begintime
from pymysql_util import get_sharp_data_records

connection = pymysql.connect(
        host='129.226.126.98',
        user='sloarflare',
        password='YMY5TJ3MXiaeFcAb',
        database='sloarflare',
        charset='utf8mb4'
    )

# connection = pymysql.connect(
#         host='localhost',
#         user='root',
#         password='wjf666',
#         database='solarflare',
#         charset='utf8mb4'
#     )



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
            if connection:
                connection.close()

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
            # if cursor:
            #     cursor.close()
            # if connection:
            #     connection.close()

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
def sharp_image_simple_getnumber_add(download_day: datetime, NOAA_ARS,begin):
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

#
if __name__ == '__main__':
    # insertnum =sharp_image_simple_download_para\
    #     (datetime.strptime("2024.11.12", "%Y.%m.%d"), 13883
    #      ,"21:59"
    #      )
    # '''
    # 13891 75
    # '''
    # insertnum = sharp_image_simple_download(datetime.strptime("2024.07.21", "%Y.%m.%d"), 13744)
    # print(insertnum)
    # begintime = calculate_begintime(105)  # begin 例如22:12 需要判断下
    # insertnum = sharp_image_simple_download_para(datetime.strptime("2024.11.12", "%Y.%m.%d"), 13883,
    #                                              begintime)
    # records = get_sharp_data_records("20241112", 13883)
    # print(len(records))
    # pretoday_datetime = datetime(2023, 5, 3)
    # date_format_tenfeature = pretoday_datetime.strftime("%Y.%m.%d")
    # insertnum, noaa_ars_para, noaa_num_para = sharp_image_simple_getnumber(datetime.strptime(date_format_tenfeature, "%Y.%m.%d"), 13289)
    # insertnum=sharp_image_simple_download_para(datetime.strptime(date_format_tenfeature, "%Y.%m.%d"), 13289,"23:47",noaa_ars_para,noaa_num_para)
    # print(insertnum)
    pass

