import ftplib
import logging
import os
import time
from datetime import timedelta,datetime
from process_util import download_file, AR_info_data_process, get_ten_feather_noaaid_information, \
    get_120_ten_feather_data, calculate_begintime, get_120_ten_feather_data_in_two_day, update_sql_and_csv
from csv_util import append_ar_flare_prediction_to_csv, append_sharp_data_to_csv
from pymysql_util import get_noaa_ids_for_today


def download_AR_information_from_ftp(year, month, day):
    """
    month:02
    day:28
    """
    url_base = "/pub/forecasts/SRS/"  # ftp://ftp.swpc.noaa.gov/pub/forecasts/SRS/
    file_name = f'{month}{day}SRS.txt'
    download_url = url_base + file_name
    file_dir_path = f'./data/Solar_AR_info/{year}/{month}/'
    download_file_path = os.path.join(file_dir_path, file_name)
    if not os.path.exists(file_dir_path):
        os.makedirs(file_dir_path)
    try:
        logging.info(f'AR_INFO {year}{month}{day} 开始下载')
        # 两次重复下载的机会
        download_code = download_file(download_url, file_dir_path, file_name)
        if download_code != 0:
            logging.warning(f"download_AR_information download_code={download_code}")
            download_code = download_file(download_url, file_dir_path, file_name)
            if download_code != 0:
                logging.error("download_AR_information错误，从FTP获取AR_INFO出错。")
                return download_code
        logging.info(f'AR_INFO {year}{month}{day} 下载完成，开始处理数据')
        ar_flare_prediction=AR_info_data_process(download_file_path, year, month, day)
        append_ar_flare_prediction_to_csv(ar_flare_prediction)
        logging.info('AR_INFO 数据处理成功')
        return 0
    except ftplib.error_temp as e:
        err_info = "{0}".format(e)
        if "No such file or directory" in err_info:
            logging.warning(f"AR_info未找到{year}{month}{day}下载文件")
            return -1
        else:
            logging.error('未知错误-999', err_info)
            if os.path.exists(download_file_path):
                os.remove(download_file_path)
            return -999
    except Exception as e:
        err_info = "{0}".format(e)
        logging.error('AR_info下载未知错误 code:-999', err_info)
        if os.path.exists(download_file_path):
            os.remove(download_file_path)
        return -999

def main(today: datetime):
    '''
        比如预报2025.08.11爆发情况，那么today就是2025.08.11
    '''
    result_today_predict_str = today.strftime("%Y%m%d")
    ar_today_datetime = today
    print("=================================第一步先获取AR信息插入到数据库，并且标记是否够60以内=================================")
    ar_year, ar_month, ar_day = \
    ar_today_datetime.strftime("%Y"), ar_today_datetime.strftime("%m"), ar_today_datetime.strftime("%d")
    download_AR_information_from_ftp(ar_year, ar_month, ar_day)
    print("=================================第二步循环遍历当天活动区ID，获取10维特征=================================")
    NOAAIDList = get_noaa_ids_for_today(ar_today_datetime)  # 获取当天的 NOAA ID 列表
    print("NOAAIDList:", NOAAIDList)  # 输出 NOAAIDList
    ten_feather_yesterday_datetime=today-timedelta(days=1) #得到昨天的日期datetime格式
    ten_feather_yesterday_str = ten_feather_yesterday_datetime.strftime("%Y.%m.%d")
    datetime_for_update_ars=datetime.strptime(f"{ar_year}-{ar_month}-{ar_day}", "%Y-%m-%d").strftime("%Y-%m-%d 00:00:00")
    ten_feather_noaaid_list=set() #按照2024.11月去重逻辑，使用集合去重，多活动区id多次出现问题
    for NOAAID in NOAAIDList:
        if NOAAID not in ten_feather_noaaid_list:
            time_step, noaaid_str, noaaid_number = get_ten_feather_noaaid_information\
                    (datetime.strptime(ten_feather_yesterday_str, "%Y.%m.%d"),
                    NOAAID)

            if type(noaaid_str) != int:
                ten_feather_noaaid_list.update(noaaid_str.split(',')) #加到集合里面

            data_to_insert=None

            if time_step == 120: #如果不用补
                insert_num,data_to_insert = get_120_ten_feather_data\
                    (datetime.strptime(ten_feather_yesterday_str, "%Y.%m.%d"),
                     NOAAID,
                     noaaid_str,
                     noaaid_number)
                print(NOAAID, "插入特征完成", fr"插入{insert_num}条")

            elif 120 > time_step >= 110:
                begintime = calculate_begintime(time_step)  # begin 例如22:12 需要判断下
                insert_num,data_to_insert = get_120_ten_feather_data_in_two_day\
                        (datetime.strptime(ten_feather_yesterday_str, "%Y.%m.%d"),
                         NOAAID,begintime,
                         noaaid_str,
                         noaaid_number)
                print(NOAAID, "补充插入特征完成", fr"插入{insert_num}条")

            else:
                insert_num=time_step
                print(NOAAID, f"不足特征完成特征只有{insert_num}条","不插入!")
            append_sharp_data_to_csv(data_to_insert)


            isin120 = f"{insert_num}"
            # 根据date_format_ar和NOAAID数值 去ar_flare_prediction查询那一行插入isin120的字符串数值在is120字段
            # 以及csv操作
            update_sql_and_csv\
                    (datetime_for_update_ars,
                    NOAAID,
                    isin120)


if __name__ == '__main__':
    start_date = datetime.today()
    start_date = datetime(2025,8,11)
    end_date = start_date
    now = time.time()
    # 打开 log.txt 文件用于写入
    with open(f".\log\{start_date.strftime('%Y-%m-%d_%H-%M-%S')}-log.txt", "a", encoding="utf-8") as log_file:
        # 创建 DualStream 实例，输出重定向到控制台和文件
        # 循环遍历日期范围
        while start_date <= end_date:
            # 调用 main 函数，传入当前日期
            print(start_date, "十点到达，开始执行脚本")
            main(start_date)
            print("***************************************************************************************************************************************")
            print(start_date, "操作结束")
            print("***************************************************************************************************************************************")
            # 日期递增 1 天
            start_date += timedelta(days=1)
