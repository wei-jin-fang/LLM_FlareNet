import json
from datetime import datetime
from datetime import datetime, timedelta
import requests
from lxml import html

from process_util import insert_event_data_to_sql_and_csv, get_data_noaaars_and_Nmbr, find_max_label_for_noaaid_in_day, \
    update_result_by_overall_max_label_and_noaaid
from tools import find_max_label_for_noaaARs_in_day

def main(today: datetime,read_mode="sql"):
    result_today_predict_str = today.strftime("%Y%m%d")

    ten_feather_yesterday_datetime = today - timedelta(days=1)  # 得到昨天的日期datetime格式
    T_REC_base_year, T_REC_base_month, T_REC_baser_day = \
        ten_feather_yesterday_datetime.strftime("%Y"), \
        ten_feather_yesterday_datetime.strftime("%m"), \
        ten_feather_yesterday_datetime.strftime("%d")
    T_REC_base = f"{T_REC_base_year}{T_REC_base_month}{T_REC_baser_day}"  # 用于查询10维度参数数据表的时间参数前缀 对应前一天
    print("T_REC_base", T_REC_base)
    ars_nmbr_results = get_data_noaaars_and_Nmbr(T_REC_base, read_mode)
    print(ars_nmbr_results)
    for key, value in ars_nmbr_results.items():
        NOAA_ARS = key
        print(
            "=================================第六步,读取Event数据更新Result的Y_true=================================")
        print("============================6.1遍历NOAA_ARS，去Event找到最大值以及对应代表ID=============================")
        print("！！！！注意，实时预报过程打标签不一定是真正的最大类别，因为此时今日预报的所有还没全部爬取到，需要模拟拿多一段时间，根据startday进行综合找到最大值！！！！！！！")
        max_goes_class_dict = {}
        noaa_ars_list = NOAA_ARS.split(',')
        # 找到每个 NOAA_ARS 对应的最大 GOES_Class 和 NOAAID
        max_goes_classes = find_max_label_for_noaaid_in_day(result_today_predict_str, noaa_ars_list,read_mode)
        print(max_goes_classes)
        # print(max_goes_classes) #
        # 比如四个里面有俩有数据的
        # {'13880': ('N0', 13880), '13883': ('C7.3', 13883), '13884': ('N0', 13884), '13886': ('C5.4', 13886)}
        # 找到所有 NOAA_ARS 中整体最大的 GOES_Class 和 NOAAID
        overall_max_goes_class, overall_max_noaa_id = find_max_label_for_noaaARs_in_day(max_goes_classes)
        print(f"整体最大 GOES_Class 为 {overall_max_goes_class}，对应 NOAAID 为 {overall_max_noaa_id}")
        update_result_by_overall_max_label_and_noaaid(result_today_predict_str, NOAA_ARS, overall_max_goes_class, overall_max_noaa_id)

if __name__ == '__main__':
    # main(start_date = datetime.today())
    main(datetime(2025,8,11),read_mode="csv")

    pass
