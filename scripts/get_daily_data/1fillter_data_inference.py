import ftplib
import logging
import os
import time
from datetime import timedelta, datetime
from process_util import download_file, AR_info_data_process, get_ten_feather_noaaid_information, \
    get_120_ten_feather_data, calculate_begintime, get_120_ten_feather_data_in_two_day, update_sql_and_csv, \
    get_data_noaaars_and_Nmbr, check_data_is_available, get_120_ten_feather_data_by_T_REC_base_and_Nmbr, \
    global_data_by_mean_and_std, inference_by_data, insert_result_data
from csv_util import append_ar_flare_prediction_to_csv, append_sharp_data_to_csv
from pymysql_util import get_noaa_ids_for_today
from tools import truncate

def main(today: datetime,read_mode="sql"):
    '''
        比如预报2025.08.11爆发情况，那么today就是2025.08.11
    '''
    result_today_predict_str = today.strftime("%Y%m%d")
    ar_today_datetime = today
    print("=================================第一步先获取AR信息插入到数据库，并且标记是否够60以内=================================")
    ar_year, ar_month, ar_day = \
        ar_today_datetime.strftime("%Y"), ar_today_datetime.strftime("%m"), ar_today_datetime.strftime("%d")
    # download_AR_information_from_ftp(ar_year, ar_month, ar_day)
    print("=================================第二步循环遍历当天活动区ID，获取10维特征=================================")
    # NOAAIDList = get_noaa_ids_for_today(ar_today_datetime)  # 获取当天的 NOAA ID 列表
    # print("NOAAIDList:", NOAAIDList)  # 输出 NOAAIDList
    ten_feather_yesterday_datetime = today - timedelta(days=1)  # 得到昨天的日期datetime格式
    ten_feather_yesterday_str = ten_feather_yesterday_datetime.strftime("%Y.%m.%d")
    datetime_for_update_ars = datetime.strptime(f"{ar_year}-{ar_month}-{ar_day}", "%Y-%m-%d").strftime(
        "%Y-%m-%d 00:00:00")

    # ten_feather_noaaid_list = set()  # 按照2024.11月去重逻辑，使用集合去重，多活动区id多次出现问题
    # for NOAAID in NOAAIDList:
    #     if NOAAID not in ten_feather_noaaid_list:
    #         time_step, noaaid_str, noaaid_number = get_ten_feather_noaaid_information \
    #             (datetime.strptime(ten_feather_yesterday_str, "%Y.%m.%d"),
    #              NOAAID)
    #
    #         if type(noaaid_str) != int:
    #             ten_feather_noaaid_list.update(noaaid_str.split(','))  # 加到集合里面
    #
    #         data_to_insert = None
    #
    #         if time_step == 120:  # 如果不用补
    #             insert_num, data_to_insert = get_120_ten_feather_data \
    #                 (datetime.strptime(ten_feather_yesterday_str, "%Y.%m.%d"),
    #                  NOAAID,
    #                  noaaid_str,
    #                  noaaid_number)
    #             print(NOAAID, "插入特征完成", fr"插入{insert_num}条")
    #
    #         elif 120 > time_step >= 110:
    #             begintime = calculate_begintime(time_step)  # begin 例如22:12 需要判断下
    #             insert_num, data_to_insert = get_120_ten_feather_data_in_two_day \
    #                 (datetime.strptime(ten_feather_yesterday_str, "%Y.%m.%d"),
    #                  NOAAID, begintime,
    #                  noaaid_str,
    #                  noaaid_number)
    #             print(NOAAID, "补充插入特征完成", fr"插入{insert_num}条")
    #
    #         else:
    #             insert_num = time_step
    #             print(NOAAID, f"不足特征完成特征只有{insert_num}条", "不插入!")
    #         append_sharp_data_to_csv(data_to_insert)
    #
    #         isin120 = f"{insert_num}"
    #         根据date_format_ar和NOAAID数值 去ar_flare_prediction查询那一行插入isin120的字符串数值在is120字段
    #         以及csv操作
            # update_sql_and_csv \
            #     (datetime_for_update_ars,
            #      NOAAID,
            #      isin120)
    print("=================================第三步,根据留下来的10维度特征，去判断他们当时使用的活动区ID是否合法，生成今天需要使用的NOAAARS=================================")
    print("=================================第三步,这个时候最后一列还没有更新，目前最后一列仅用于筛选是否提取出啦120个特征=================================")
    T_REC_base_year, T_REC_base_month, T_REC_baser_day = \
        ten_feather_yesterday_datetime.strftime("%Y"),\
        ten_feather_yesterday_datetime.strftime("%m"), \
        ten_feather_yesterday_datetime.strftime("%d")

    T_REC_base = f"{T_REC_base_year}{T_REC_base_month}{T_REC_baser_day}"  # 用于查询10维度参数数据表的时间参数前缀 对应前一天
    print("T_REC_base",T_REC_base)

    # 生成今天数据的NOAA_ARS和Nmbr。根据T_REC_base这个变量，对于sharp_data_ten_feature进行查询T_REC字段,只要是以T_REC_base变量开头就行，
    # 然后根据查询到的NOAA_ARS进行分组，然后拿到这个日期下的NOAA_ARS和对应的Nmbr字段对应，存到一个集合里面，key是NOAA_ARS
    ars_nmbr_results = get_data_noaaars_and_Nmbr(T_REC_base,read_mode)
    print(ars_nmbr_results) # 例如：{'13878,13879': {'13878'}, '13880,13883,13884,13886': {'13886'},
    print("=================================第四步,爬取Event插入到数据库，但是每日预报虽然预报今天发生，但是只能拿到今天数据填充=================================")

    print("=================================第五步,依次读取每一张图数据，单个或者多个活动区数据，整理，进行模型推理=================================")
    for key, value in ars_nmbr_results.items():
        NOAA_ARS = key
        Nmbr = list(value)[0]  # 插入时候代表ID 【0】是因为数据格式
        # 判断是否在60是否满足120的要求的字段 ：
        # datetime_for_update_ars和 Nmbr(获取特征时候最后一列那个留下来的活动区ID) 去 ar_flare_prediction 判断is60是否是T，is120字段是否是120、
        is_valid = check_data_is_available(datetime_for_update_ars, Nmbr,read_mode)
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
                    print(T_REC_base,Nmbr)#20240101
                    records = get_120_ten_feather_data_by_T_REC_base_and_Nmbr\
                        (T_REC_base,
                         Nmbr,
                         read_mode)

                    # 转换成双重列表
                    list_data = [list(inner_tuple) for inner_tuple in records]
                    print(list_data)
                    # 接下来进行归一化
                    globdata = global_data_by_mean_and_std(list_data)
                    # 取出120/3 NN也是取出三个一步
                    globdata = globdata[::3]

                    # 接下来进行模型推理,拿到10个概率矩阵,监测类型，模型
                    probabilitiesList = inference_by_data([_model_type], globdata, JianceType)
                    '''
                        接下来整理数据结构，写入数据
                    '''

                    insertdata = []
                    insertdata.extend([T_REC_base, _model_type, JianceType, NOAA_ARS])
                    insertdata.extend(probabilitiesList)
                    insertdata.extend([Nmbr])
                    insertdata.extend([truncate(sum(probabilitiesList) / len(probabilitiesList), 3)])
                    # 添加代表ID和y_true
                    insertdata.extend([""]) #这时候没有全局最大值类别，设置为""
                    insertdata.extend([""]) #这时候没有全局最大值类别对应的id，设置为""
                    insertdata.extend([str(result_today_predict_str)]) #预报日期，数据日期的后一天
                    insert_result_data(insertdata,read_mode)

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
                    insertdata.extend([str(result_today_predict_str)])
                    insert_result_data(insertdata,read_mode)


if __name__ == '__main__':
    # start_date = datetime.today()
    start_date = datetime(2025,8,11)
    start_date-=timedelta()
    end_date = start_date
    now = time.time()
    # 打开 log.txt 文件用于写入
    with open(f".\log\{start_date.strftime('%Y-%m-%d_%H-%M-%S')}-log.txt", "a", encoding="utf-8") as log_file:
        # 创建 DualStream 实例，输出重定向到控制台和文件
        # 循环遍历日期范围
        while start_date <= end_date:
            # 调用 main 函数，传入当前日期
            print(start_date, "十点到达，开始执行脚本")
            main(start_date,"csv")
            print(
                "****************************************************************************")
            print(start_date, "操作结束")
            print(
                "****************************************************************************")
            # 日期递增 1 天
            start_date += timedelta(days=1)

