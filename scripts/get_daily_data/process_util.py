import logging
import shutil
import time
import uuid
from decimal import Decimal
from ftplib import FTP
from datetime import datetime, timedelta

import pymysql
import requests
import urllib3

from pymysql_util import insert_ar_info, connection, update_is120_field,check_is60_is120
import os

from csv_util import update_is120_in_csv, check_data_in_csv


def check_data_is_available(date_format_ar, NOAAID):
    print(check_is60_is120(date_format_ar, NOAAID))
    return (check_data_in_csv(date_format_ar, NOAAID))

def get_data_noaaars_and_Nmbr(T_REC_base):
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
        cursor.execute(query_sql, (f"{T_REC_base}%",))  #因为之前就算补充的话拼接了下载当前的，所有通配符可以模糊查询
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

def update_sql_and_csv(SELECT_date_format_ar,NOAAID,isin120):
    update_is120_field(SELECT_date_format_ar, NOAAID, isin120)
    update_is120_in_csv(SELECT_date_format_ar, NOAAID, isin120)

def get_120_ten_feather_data_in_two_day(download_day: datetime, NOAA_ARS,begin,noaa_ars_para,noaa_num_para):
    """
    :param download_datetime: 下载的图像的时间点
    """
    previous_day = download_day - timedelta(days=1)  #在往前一天，因为要补充
    res = None
    Columns = ["TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP",
               "USFLUX", "AREA_ACR", "MEANPOT", "R_VALUE", "SHRGT45"]

    resultstr = ",".join(Columns)
    try:
        DATA_SERIES = "hmi.sharp_cea_720s_nrt"
        download_url = "http://jsoc.stanford.edu/cgi-bin/ajax/jsocextinfo"
        download_param = {
            # previous_day 和 download_day
            "ds": f'{DATA_SERIES}[][{previous_day.strftime("%Y.%m.%d")}_{begin}_TAI-{download_day.strftime("%Y.%m.%d")}_23:48_TAI][? NOAA_ARS ~ "{NOAA_ARS}"  ?]'+ '{magnetogram}',
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
                # 拼接download_day到previous_day之前，到时候方便提取特征，否则有两个日期
                record_time_list[i] = f"{download_day.strftime('%Y%m%d')}_{record_time[:]}"
            # 不区分所有的都赋值那天主要的，不管是不是补的，反正要保证补的和实际那天一样
            noaa_num_list[i] = noaa_num_para
            noaa_ars_list[i] = noaa_ars_para

        TOTUSJH_list = download_datas['keywords'][4]['values']
        TOTPOT_list = download_datas['keywords'][5]['values']
        TOTUSJZ_list = download_datas['keywords'][6]['values']
        ABSNJZH_list = download_datas['keywords'][7]['values']
        SAVNCPP_list = download_datas['keywords'][8]['values']
        USFLUX_list = download_datas['keywords'][9]['values']
        AREA_ACR_list = download_datas['keywords'][10]['values']
        MEANPOT_list = download_datas['keywords'][11]['values']
        R_VALUE_list = download_datas['keywords'][12]['values']
        SHRGT45_list = download_datas['keywords'][13]['values']
        insert_num=len(SHRGT45_list)
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
                    '''
                        注释了统一使用联合主键去重
                    '''
                    # result = cursor.fetchone()
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

        return insert_num,data_to_insert
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

def calculate_begintime(insertnum):
    # 基础的结束时间是23:59
    end_time = datetime.strptime("23:59", "%H:%M")

    # 每个时间块的时长（比如每个数据块占12分钟）
    time_block_duration = timedelta(minutes=12)

    # 计算距离23:59的时间差
    remaining_blocks = 120 - insertnum  # 剩余需要下载的块数

    # 计算开始时间
    begintime = end_time - time_block_duration * remaining_blocks

    return begintime.strftime("%H:%M")

def get_120_ten_feather_data(download_day: datetime, NOAA_ARS,noaa_ars_para,noaa_num_para):
    """
    :param download_datetime: 下载的图像的时间点
    """
    res = None
    Columns = ["TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP",
               "USFLUX", "AREA_ACR", "MEANPOT", "R_VALUE", "SHRGT45"]
    resultstr = ",".join(Columns)
    try:
        DATA_SERIES = "hmi.sharp_cea_720s_nrt"
        download_url = "http://jsoc.stanford.edu/cgi-bin/ajax/jsocextinfo"
        download_param = {
            # 1d 就直接一天的意思
            "ds": f'{DATA_SERIES}[][{download_day.strftime("%Y.%m.%d")}/1d][? NOAA_ARS ~ "{NOAA_ARS}"  ?]'+ '{magnetogram}',
            "op": "rs_list",
            "key": "T_REC,HARPNUM,NOAA_NUM,NOAA_ARS,"+resultstr, #NOAA_NUM有几个活动区  NOAA_ARS（个数与NOAA_NUM对应）
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

        for i in range(len(record_time_list)):
            # 不区分所有的都赋值成那天主要的，不管是不是补的，反正要保证补的和实际那天的活动区id字符串和活动区数量一样
            noaa_num_list[i] = noaa_num_para
            noaa_ars_list[i] = noaa_ars_para

        TOTUSJH_list = download_datas['keywords'][4]['values']
        TOTPOT_list = download_datas['keywords'][5]['values']
        TOTUSJZ_list = download_datas['keywords'][6]['values']
        ABSNJZH_list = download_datas['keywords'][7]['values']
        SAVNCPP_list = download_datas['keywords'][8]['values']
        USFLUX_list = download_datas['keywords'][9]['values']
        AREA_ACR_list = download_datas['keywords'][10]['values']
        MEANPOT_list = download_datas['keywords'][11]['values']
        R_VALUE_list = download_datas['keywords'][12]['values']
        SHRGT45_list = download_datas['keywords'][13]['values']
        insert_num=len(SHRGT45_list)
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

                dataorder = 1  #为了保证补充的在前面，因为日期本身就在前面
                for record_time, har_num, noaa_num, noaa_ars, totusjh, totpot, totusjz, absnjzh, savncpp, usflux, area_acr, meanpot, r_value, shrgt45 in zip(
                        record_time_list, HARPNUM_list, noaa_num_list, noaa_ars_list,
                        TOTUSJH_list, TOTPOT_list, TOTUSJZ_list, ABSNJZH_list,
                        SAVNCPP_list, USFLUX_list, AREA_ACR_list, MEANPOT_list,
                        R_VALUE_list, SHRGT45_list):
                    '''
                        现在注释掉了统一从数据库联合主键层面去重
                    '''
                    # cursor.execute(select_sql, (record_time, noaa_ars))
                    # result = cursor.fetchone()
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
            if cursor:
                cursor.close()

        return insert_num,data_to_insert
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

def get_ten_feather_noaaid_information(download_day: datetime, NOAA_ARS):
    """
    :param download_datetime: 下载的图像的时间点
    """
    res = None
    Columns = ["TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP",
               "USFLUX", "AREA_ACR", "MEANPOT", "R_VALUE", "SHRGT45"]
    resultstr = ",".join(Columns)
    try:
        DATA_SERIES = "hmi.sharp_cea_720s_nrt"
        download_url = "http://jsoc.stanford.edu/cgi-bin/ajax/jsocextinfo"
        download_param = {
            # 判断下载日期的那天的磁图数量和10维特征的数量 0点到23点48分钟，12分钟一张
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

        noaa_ars_list = download_datas['keywords'][3]['values']
        SHRGT45_list = download_datas['keywords'][13]['values']
        insertnum=len(SHRGT45_list)#算下有多少张

        numbers_set = set()
        # 遍历 noaa_ars_list 中的每个字符串
        for item in noaa_ars_list:
            # 分割每个字符串，提取数字，并加入集合
            numbers_set.update(item.split(','))   #因为互动区出来好几次，去重，然后把所有重复的去掉就是这一天出现的活动区编号，种类（不重复）
        # 将集合中的数字按逗号拼接成一个新的字符串
        result_string = ','.join(sorted(numbers_set, key=int)) #恢复10388,10399这样子格式

        return insertnum,result_string,len(numbers_set) #算一下活动区有几个
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
    return AR_info_list

def download_file(ftp_file_path, dst_file_path, save_file_name):
    """
    从ftp下载文件到本地
    :param ftp_file_path: ftp下载文件路径
    :param dst_file_path: 本地存放路径
    :param save_file_name: 保存文件名
    :return:
    """
    if os.path.exists(os.path.join(dst_file_path, save_file_name)):
        logging.info(f"{save_file_name}已存在，不再重复下载")
        return 0
    ftp = FTP()
    ftp.set_debuglevel(0)
    try:
        ftp.connect(host="ftp.swpc.noaa.gov", port=21)  # 连接FTP
        ftp_user = ""
        ftp_password = ""
        ftp.login(ftp_user, ftp_password)  # FTP登录
        ftp.set_pasv(True)  # 设置FTP为被动模式
        buffer_size = 8192  # 设置缓冲区大小，默认是8192

        file_list = ftp.nlst(ftp_file_path)  # 获取文件列表
        for file_name in file_list:
            ftp_file = os.path.join(ftp_file_path, file_name)
            write_file = os.path.join(dst_file_path, save_file_name + '.download')
            save_file = os.path.join(dst_file_path, save_file_name)
            with open(write_file, "wb") as f:
                ftp.retrbinary('RETR %s' % ftp_file, f.write, buffer_size)
            shutil.move(str(write_file), str(save_file))
        return 0
    except Exception as e:
        if str(e).startswith('550'):
            print(f'文件不存在于FTP服务器: {save_file_name}')
            return -1
        else:
            logging.info(f'下载{save_file_name}出错，错误日志：{e}')
            return -999
    finally:
        ftp.quit()  # 关闭FTP连接，释放资源