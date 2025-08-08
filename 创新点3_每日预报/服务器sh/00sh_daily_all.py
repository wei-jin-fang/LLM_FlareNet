# 下载太阳活动区信息
'''
https://chatgpt.com/c/672b5a87-76d8-8000-b4f7-67f7db147a61
'''
import ftplib
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from ftplib import FTP
from pymysql_util import db_update, get_noaa_ids_for_today, insert_ar_info, update_is120_field, check_is60_is120, \
    get_sharp_data_records, fetch_noaa_ars_nmbr, insert_forecast_data, insert_event_data, \
    find_max_goes_class_for_noaa_ars,AR_info_data_process
from dataprocess import Globaldata, find_overall_max_goes_class, calculate_begintime, DualStream
from tools import  truncate
from pymysql_util import sharp_image_simple_download, sharp_image_simple_getnumber, sharp_image_simple_download_para
from datetime import datetime
from run_main_Test_sh import mainTestdataData

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

def download_AR_information(year, month, day):
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
        AR_info_data_process(download_file_path, year, month, day)
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
    # https://chatgpt.com/c/672b5a87-76d8-8000-b4f7-67f7db147a61
    # 获取昨天天的日期
    '''
    下面print以today20240102为例子
    '''

    date_format_event = today.strftime("%Y%m%d")  # 用于查询event数据
    date_format_ar = today.strftime("%Y%m%d")  # 用于下载ar数据的时间参数 ,用于查询ar数据表的时间参数
    ar_day_datetime = today

    pretoday_datetime = today - timedelta(1)#拿到第二天的活动区ID，往前一天对应需要得数据那天
    date_format_tenfeature = pretoday_datetime.strftime("%Y.%m.%d")  # 用于下载10维特征的时间参数参数

    print("=================================第一步先获取AR信息插入到数据库，并且标记是否够60以内=================================")

    ar_year, ar_month, ar_day = ar_day_datetime.strftime("%Y"), ar_day_datetime.strftime("%m"), ar_day_datetime.strftime("%d")
    # print(ar_year, ar_month, ar_day)2024 01 02
    start_time = datetime.strptime(date_format_ar, "%Y%m%d")
    end_time = datetime.strptime(date_format_ar, "%Y%m%d")
    while start_time <= end_time:
        #break;
        print(start_time.strftime("%Y"), start_time.strftime("%m"), start_time.strftime("%d"))
        download_AR_information(start_time.strftime("%Y"), start_time.strftime("%m"), start_time.strftime("%d"))
        start_time = start_time + timedelta(days=1)
    print("=================================第二步循环遍历当天活动区ID，获取10维特征=================================")
    # print(ar_day_datetime)2024-01-02 00:00:00
    NOAAIDList = get_noaa_ids_for_today(ar_day_datetime)  # 获取当天的 NOAA ID 列表
    print("NOAAIDList:", NOAAIDList)  # 输出 NOAAIDList
    SELECT_date_format_ar = datetime.strptime(f"{ar_year}-{ar_month}-{ar_day}", "%Y-%m-%d").strftime("%Y-%m-%d 00:00:00")
    # print(SELECT_date_format_ar)2024-01-02 00:00:00
    thisday_ars_all=set()
    for NOAAID in NOAAIDList:
        #break;
        if NOAAID not in thisday_ars_all:
            insertnum,noaa_ars_para,noaa_num_para = sharp_image_simple_getnumber(datetime.strptime(date_format_tenfeature, "%Y.%m.%d"), NOAAID)
            if type(noaa_ars_para) != int:
                thisday_ars_all.update(noaa_ars_para.split(',')) #加到集合里面
            if insertnum == 120:
                insertnum = sharp_image_simple_download(datetime.strptime(date_format_tenfeature, "%Y.%m.%d"), NOAAID,noaa_ars_para,noaa_num_para)
                print(NOAAID, "插入特征完成", fr"插入{insertnum}条")
            elif 120 > insertnum >= 110:
                begintime = calculate_begintime(insertnum)  # begin 例如22:12 需要判断下
                insertnum = sharp_image_simple_download_para(
                    datetime.strptime(date_format_tenfeature, "%Y.%m.%d"), NOAAID,begintime,noaa_ars_para,noaa_num_para)
                print(NOAAID, "补充插入特征完成", fr"插入{insertnum}条")
                # add_insertnumber=sharp_image_simple_getnumber_add(datetime.strptime(date_format_tenfeature, "%Y.%m.%d"), NOAAID,begintime)
                # if add_insertnumber==120:
                #     insertnum = sharp_image_simple_download_para(datetime.strptime(date_format_tenfeature, "%Y.%m.%d"), NOAAID,begintime)
                #     print(NOAAID, "补充插入特征完成", fr"插入{insertnum}条")
                # else:
                #     insertnum = sharp_image_simple_download(datetime.strptime(date_format_tenfeature, "%Y.%m.%d"), NOAAID)
                #     print(NOAAID, "插入特征补充了也不够", fr"插入{insertnum}条")
            else:
                insertnum = sharp_image_simple_download(datetime.strptime(date_format_tenfeature, "%Y.%m.%d"), NOAAID,noaa_ars_para,noaa_num_para)
                print(NOAAID, "不足，插入特征完成", fr"插入{insertnum}条")
            isin120 = ""
            if insertnum == 120:
                isin120 = "120"
            else:
                isin120 = f"{insertnum}"
            # 根据date_format_ar和NOAAID数值 去ar_flare_prediction查询那一行插入isin120的字符串数值在is120字段
            affected_rows = update_is120_field(SELECT_date_format_ar, NOAAID, isin120)
            pass
    print("=================================第三步,根据留下来的10维度特征，去判断他们当时使用的活动区ID是否合法，生成今天需要使用的NOAAARS=================================")
    print("=================================第三步,这个时候最后一列还没有更新，目前最后一列仅用于筛选是否提取出啦120个特征=================================")
    T_REC_base_year, T_REC_base_month, T_REC_baser_day = pretoday_datetime.strftime("%Y"), pretoday_datetime.strftime("%m"), pretoday_datetime.strftime("%d")
    T_REC_base = f"{T_REC_base_year}{T_REC_base_month}{T_REC_baser_day}"  # 用于查询10维度参数数据表的时间参数前缀 对应前一天
    print("T_REC_base",T_REC_base)

    # 生成今天数据的NOAA_ARS和Nmbr。根据T_REC_base这个变量，对于sharp_data_ten_feature进行查询T_REC字段,只要是以T_REC_base变量开头就行，
    # 然后根据查询到的NOAA_ARS进行分组，然后拿到这个日期下的NOAA_ARS和对应的Nmbr字段对应，存到一个集合里面，key是NOAA_ARS
    ars_nmbr_results = fetch_noaa_ars_nmbr(T_REC_base)
    print(ars_nmbr_results)
    # # {'13878,13879': {'13878'}, '13880,13883,13884,13886': {'13886'},
    # #  '13862,13863,13865,13866,13868,13869,13870,13871,13872,13873,13876': {'13869'}, '13881': {'13881'}}
    print("=================================第四步,爬取Event插入到数据库，但是每日预报虽然预报今天发生，但是只能拿到今天数据填充=================================")
    from getEvent import getTodayEvent
    EventdataList = getTodayEvent()
    insert_event_data(EventdataList)
    print("=================================第五步,依次读取每一张图数据，单个或者多个活动区数据，整理，进行模型推理=================================")
    for key, value in ars_nmbr_results.items():
        # print(list(value)[0])

        NOAA_ARS = key
        Nmbr = list(value)[0]  # 插入时候代表ID 【0】是因为数据格式
        # 判断是否在60是否满足120的要求的字段 ：SELECT_date_format_ar和 Nmbr(获取特征时候最后一列那个留下来的活动区ID) 去 ar_flare_prediction 判断is60是否是T，is120字段是否是120、
        # print(date_format_ar)20240102
        is_valid = check_is60_is120(date_format_ar, Nmbr)
        print(Nmbr, is_valid)
        for _model_type in ["LLM_VIT"]:
            for JianceType in ["TSS"]:
                if is_valid:
                    '''
                    获取当前活动区对应的120数据
                    根据 T_REC_base变量和NOAAID 变量，对应去查询sharp_data_ten_feature表的对应 T_REC字段（只要以T_REC_base变量变量开头视为匹配）和Nmbr字段进行拿出记录
                    然后根据T_REC字段的数据进行升序，即时时间递增的顺序，拿到120条记录，把其中的 ["NOAA_AR", "T_REC", "NOAA_NUM", "CLASS", "TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP",
                   "USFLUX", "AREA_ACR", "MEANPOT", "R_VALUE", "SHRGT45"]提取出来，记住顺序不要乱，因为我是时序数据。存到一个二维list；里面
                    '''
                    print(T_REC_base)#20240101
                    print((Nmbr))
                    records = get_sharp_data_records(T_REC_base, Nmbr)
                    print(len(records))  #14
                    # 转换成双重列表
                    list_data = [list(inner_tuple) for inner_tuple in records]

                    # 接下来进行归一化
                    globdata = Globaldata(list_data)
                    # 取出120/3 NN也是取出三个一步
                    globdata = globdata[::3]

                    #
                    # 接下来进行模型推理,拿到10个概率矩阵,监测类型，模型
                    probabilitiesList = mainTestdataData([_model_type], globdata, JianceType)

                    print(
                        "=================================第六步,读取Event数据更新Result的Y_true=================================")
                    print("============================6.1遍历NOAA_ARS，去Event找到最大值以及对应代表ID=============================")
                    print("！！！！注意，实时预报过程打标签不一定是真正的最大类别，因为此时今日预报的所有还没全部爬取到，需要模拟拿多一段时间，根据startday进行综合找到最大值！！！！！！！")
                    max_goes_class_dict = {}
                    noaa_ars_list = NOAA_ARS.split(',')
                    # 找到每个 NOAA_ARS 对应的最大 GOES_Class 和 NOAAID
                    max_goes_classes = find_max_goes_class_for_noaa_ars(date_format_event, noaa_ars_list)
                    print(max_goes_classes)
                    # print(max_goes_classes) #
                    # 比如四个里面有俩有数据的
                    # {'13880': ('N0', 13880), '13883': ('C7.3', 13883), '13884': ('N0', 13884), '13886': ('C5.4', 13886)}
                    # 找到所有 NOAA_ARS 中整体最大的 GOES_Class 和 NOAAID
                    overall_max_goes_class, overall_max_noaa_id = find_overall_max_goes_class(max_goes_classes)
                    print(f"整体最大 GOES_Class 为 {overall_max_goes_class}，对应 NOAAID 为 {overall_max_noaa_id}")

                    insertdata = []
                    insertdata.extend([T_REC_base, _model_type, JianceType, NOAA_ARS])
                    insertdata.extend(probabilitiesList)
                    insertdata.extend([Nmbr])
                    insertdata.extend([truncate(sum(probabilitiesList) / len(probabilitiesList), 3)])
                    # 添加代表ID和y_true
                    insertdata.extend([str(overall_max_goes_class)[:1]])
                    insertdata.extend([str(overall_max_noaa_id)])
                    insertdata.extend([str(date_format_event)])
                    insert_forecast_data(insertdata)

                else:
                    insertdata = []
                    insertdata.extend([T_REC_base, _model_type, JianceType, NOAA_ARS])
                    insertdata.extend([-1, -1, -1, -1, -1,
                                       -1, -1, -1, -1, -1
                                       ])
                    insertdata.extend([Nmbr])
                    insertdata.extend([-1])

                    insertdata.extend([""])
                    insertdata.extend([""])
                    insertdata.extend([str(date_format_event)])
                    insert_forecast_data(insertdata)


if __name__ == '__main__':
    start_date = datetime.today()

    end_date = start_date
    now = time.time()
    # 打开 log.txt 文件用于写入
    with open(f".\log\{start_date.strftime('%Y-%m-%d_%H-%M-%S')}-log.txt", "a", encoding="utf-8") as log_file:
        # 创建 DualStream 实例，输出重定向到控制台和文件
        dual_stream = DualStream(sys.stdout, log_file)

        # 临时将 sys.stdout 重定向到 dual_stream
        sys.stdout = dual_stream

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
