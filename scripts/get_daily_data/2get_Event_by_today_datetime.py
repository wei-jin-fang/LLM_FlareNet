import json
from datetime import datetime
from datetime import datetime, timedelta
import requests
from lxml import html

from scripts.get_daily_data.process_util import insert_event_data_to_sql_and_csv


def get_today_latest_event():
    # 请求页面内容
    url = "https://www.lmsal.com/solarsoft/last_events/"
    response = requests.get(url)

    # 将响应内容转换为 HTML 树
    tree = html.fromstring(response.content)

    # 定位表格的所有行
    rows = tree.xpath('//table//tr')

    # 遍历行并提取数据
    data = []
    for row in rows:
        # 提取每一行中的所有单元格
        cells = row.xpath('td')

        # 过滤掉非事件行（如果没有足够的单元格）
        if len(cells) >= 7:

            # 提取每个单元格的文本内容并清理空白
            row_data = [cell.text_content().strip() for cell in cells]
            if (str(row_data[1]).startswith("gev")):
                data.append(row_data)
    return data

def get_event_today_and_pre_day_urlcontent(start_date):
    # with open('../创新点整理版本/创新点3_每日预报/tool/Event_data.json', 'r') as file:
    #     loaded_data = json.load(file)
    # print(loaded_data)
    # 请求页面内容

    # 将 start_date 转换为字符串格式 YYYYMMDD
    target_date = start_date
    # 计算前一天的日期
    prev_date = (datetime.strptime(start_date, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')

    # 请求页面内容
    url = "https://www.lmsal.com/solarsoft/latest_events_archive.html"
    response = requests.get(url)

    # 将响应内容转换为 HTML 树
    tree = html.fromstring(response.content)

    # 定位表格的所有行
    rows = tree.xpath('//table//tr')

    # 初始化返回值
    target_value = None
    prev_value = None

    # 遍历行并提取数据
    for row in rows:
        # 提取每一行中的所有单元格
        cells = row.xpath('td')

        # 过滤掉非事件行（如果没有足够的单元格）
        if len(cells) >= 7:
            # 提取每个单元格的文本内容并清理空白
            row_data = [cell.text_content().strip() for cell in cells]
            if str(row_data[0]).startswith("Snapshot"):
                continue

            # 解析日期并格式化
            date_obj = datetime.strptime(str(row_data[0]), '%d-%b-%Y %H:%M')
            date_key = date_obj.strftime('%Y%m%d')
            date_value = date_obj.strftime('%H%M')

            # 检查是否匹配目标日期
            if date_key == target_date:
                target_value = date_value
            # 检查是否匹配前一天
            if date_key == prev_date:
                prev_value = date_value
                break

    return prev_value, target_value


def get_event_data_by_urlcontent(year,start_date_str,start_date_event_key):


    url = f"https://www.lmsal.com/solarsoft/ssw/last_events-{year}/last_events_{start_date_str}_{start_date_event_key}/index.html"
    response = requests.get(url)

    # 将响应内容转换为 HTML 树
    tree = html.fromstring(response.content)

    # 定位表格的所有行
    rows = tree.xpath('//table//tr')

    # 遍历行并提取数据
    data = []
    for row in rows:
        # 提取每一行中的所有单元格
        cells = row.xpath('td')

        # 过滤掉非事件行（如果没有足够的单元格）
        if len(cells) >= 7:
            # 提取每个单元格的文本内容并清理空白
            row_data = [cell.text_content().strip() for cell in cells]
            if (str(row_data[1]).startswith("gev")):
                data.append(row_data)
    return data

def main(start_date=None):
    print("现在是补充时间,对于当日情况，当日的表还没有形成，肯定要拿前一天的")
    start_date = (start_date - timedelta(days=1))  #今天的时候list里面最新的是前一天的

    start_date_str = start_date.strftime("%Y%m%d")
    year = start_date.year

    pre_date_event_key,start_date_event_key  = (get_event_today_and_pre_day_urlcontent(start_date_str))
    # {'20250429': '2359', '20250428': '2359'}
    print(start_date_event_key, pre_date_event_key)
    today_event_data_list=(get_event_data_by_urlcontent(year, start_date_str, start_date_event_key))
    insert_event_data_to_sql_and_csv(today_event_data_list)

if __name__ == '__main__':
    # main(start_date = datetime.today())
    main(start_date = datetime(2025,8,12))
    # with open('../创新点整理版本/创新点3_每日预报/tool/Event_data.json', 'r') as file:
    #     loaded_data = json.load(file)
    # # 定义日期范围
    # # 20230422-20230721
    # start_date = datetime(2024, 11, 12)
    # end_date = datetime(2024, 11, 12)
    #
    # current_date = start_date
    #
    # while current_date <= end_date:
    #     # 格式化日期为所需的字符串格式
    #     date_str = current_date.strftime("%Y%m%d")
    #
    #     # 调用函数
    #     EventdataList = getEventdayurl(loaded_data, date_str)
    #     insert_event_data(EventdataList)
    #
    #     # 增加一天
    #     current_date += timedelta(days=1)
    pass
    # print(getTodayEvent())#格式【【a,b,c,d,....】，【】，【】】

# def getEventalldata():
#     # with open('../创新点整理版本/创新点3_每日预报/tool/Event_data.json', 'r') as file:
#     #     loaded_data = json.load(file)
#     # print(loaded_data)
#     # 请求页面内容
#
#
#     url = "https://www.lmsal.com/solarsoft/latest_events_archive.html"
#     response = requests.get(url)
#
#     # 将响应内容转换为 HTML 树
#     tree = html.fromstring(response.content)
#
#     # 定位表格的所有行
#     rows = tree.xpath('//table//tr')
#
#     # 遍历行并提取数据
#     data = {}
#     for row in rows:
#         # 提取每一行中的所有单元格
#         cells = row.xpath('td')
#
#         # 过滤掉非事件行（如果没有足够的单元格）
#         if len(cells) >= 7:
#
#             # 提取每个单元格的文本内容并清理空白
#             row_data = [cell.text_content().strip() for cell in cells]
#             if (str(row_data[0]).startswith("Snapshot")):
#                 continue
#             date_obj = datetime.strptime(str(row_data[0]), '%d-%b-%Y %H:%M')
#             date_obj = date_obj.strftime('%Y%m%d_%H%M')
#             data_key=date_obj[:8]
#             data_value=date_obj[-4:]
#             # Check if the date matches the target date
#             data[data_key]=data_value
#     return data
#
# # https://www.lmsal.com/solarsoft/last_events/
#
#
# def getEventdayurl(loaded_data,dateformat):
#
#     # 请求页面内容
#     # 请求页面内容
#     year=dateformat[:4]
#     urlpattern=loaded_data[dateformat]
#     url = f"https://www.lmsal.com/solarsoft/ssw/last_events-{year}/last_events_{dateformat}_{urlpattern}/index.html"
#     response = requests.get(url)
#
#     # 将响应内容转换为 HTML 树
#     tree = html.fromstring(response.content)
#
#     # 定位表格的所有行
#     rows = tree.xpath('//table//tr')
#
#     # 遍历行并提取数据
#     data = []
#     for row in rows:
#         # 提取每一行中的所有单元格
#         cells = row.xpath('td')
#
#         # 过滤掉非事件行（如果没有足够的单元格）
#         if len(cells) >= 7:
#             # 提取每个单元格的文本内容并清理空白
#             row_data = [cell.text_content().strip() for cell in cells]
#             if (str(row_data[1]).startswith("gev")):
#                 data.append(row_data)
#     return data
