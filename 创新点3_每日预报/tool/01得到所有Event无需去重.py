import json
import time
from datetime import datetime
from datetime import datetime, timedelta
import requests
from lxml import html

from 获取数据集.utils.pymysql_util import insert_event_data_back


def getEventdata():
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

def getEventalldata():
    with open('Event_data.json', 'r') as file:
        loaded_data = json.load(file)
    print(loaded_data)
    # 请求页面内容
    url = "https://www.lmsal.com/solarsoft/latest_events_archive.html"
    response = requests.get(url)

    # 将响应内容转换为 HTML 树
    tree = html.fromstring(response.content)

    # 定位表格的所有行
    rows = tree.xpath('//table//tr')

    # 遍历行并提取数据
    data = {}
    for row in rows:
        # 提取每一行中的所有单元格
        cells = row.xpath('td')

        # 过滤掉非事件行（如果没有足够的单元格）
        if len(cells) >= 7:

            # 提取每个单元格的文本内容并清理空白
            row_data = [cell.text_content().strip() for cell in cells]
            if (str(row_data[0]).startswith("Snapshot")):
                continue
            date_obj = datetime.strptime(str(row_data[0]), '%d-%b-%Y %H:%M')
            date_obj = date_obj.strftime('%Y%m%d_%H%M')
            data_key=date_obj[:8]
            data_value=date_obj[-4:]
            data[data_key]=data_value
    return data


def getEventdayurl(loaded_data,dateformat):

    # 请求页面内容
    # 请求页面内容
    year=dateformat[:4]
    urlpattern = loaded_data.get(dateformat)
    if urlpattern is not None:
        url = f"https://www.lmsal.com/solarsoft/ssw/last_events-{year}/last_events_{dateformat}_{urlpattern}/index.html"
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
    else:
        return None

if __name__ == '__main__':
    with open('Event_data.json', 'r') as file:
        loaded_data = json.load(file)
    # 定义日期范围
    # # 20230422-20230721
    # # start_date = datetime(2023, 4, 20)
    # # 2023-08-26 00:00:00 ok
    # # Traceback (most recent call last):
    # start_date = datetime(2023, 8, 27)
    # # 2023-08-27 00:00:00 None
    # # 2023-08-28 00:00:00 ok
    #
    #
    # end_date = datetime(2024, 7, 23)
    # # 2024-07-23 00:00:00 ok
    # #
    # # Process finished with exit code 0

    # start_date = datetime(2024, 11, 8)
    # # 2023-08-26 00:00:00 ok
    # # Traceback (most recent call last):
    # end_date = datetime(2024, 11, 11)
    # current_date = start_date

    '''
        这里的日期是，对应event总表的每一个日期，对应一个页面，补充下丢失的这部分，
        虽然预报日期从0423开始但是 往前几天说不定有startday是23号的活动区数据
    '''
    start_date = datetime(2023, 4, 21)
    # 2023-08-26 00:00:00 ok
    # Traceback (most recent call last):
    end_date = datetime(2023, 6, 8)
    current_date = start_date
    while current_date <= end_date:
        # 格式化日期为所需的字符串格式
        date_str = current_date.strftime("%Y%m%d")
        # 调用函数
        EventdataList = getEventdayurl(loaded_data, date_str)
        if EventdataList !=None:
            insert_event_data_back(EventdataList)
            print(current_date, "ok")
        else:
            print(current_date, "None")
        # 增加一天
        current_date += timedelta(days=1)

        time.sleep(2)
