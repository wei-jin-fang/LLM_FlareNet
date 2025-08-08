import datetime

import pymysql
'''
D:\Anaconda\envs\LLM\python.exe E:/conda_code_tf/LLM/LLM_VIT/创新点整理版本/创新点3_每日预报/服务器sh/测试本地到服务器数据库.py
数据库连接成功！
数据库版本: 8.0.36
数据库中的表: (('ar_flare_prediction',), ('eventdata',), ('result_forecast',), ('sharp_data_ten_feature',))
连接已关闭。


'''
db_config = {
        'host': '129.226.126.98',  # 数据库地址
        'user': 'sloarflare',  # 数据库用户名
        'password': 'YMY5TJ3MXiaeFcAb',  # 数据库密码
        'database': 'sloarflare',  # 数据库名
        'charset': 'utf8mb4'  # 字符集
    }


def test_db_connection():
    try:
        connection = pymysql.connect(**db_config)
        print("数据库连接成功！")

        with connection.cursor() as cursor:
            # 查询数据库版本
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            print(f"数据库版本: {version[0]}")

            # 可选：列出所有表
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            print("数据库中的表:", tables)

        connection.close()
        print("连接已关闭。")

    except pymysql.Error as e:
        print(f"数据库连接失败: {e}")

def check_is60_is120(date_format_ar, NOAAID):
    """
    查询ar_flare_prediction表中指定日期和NOAAID的记录，检查is60是否为'T'且is120是否为'120'。
    :param date_format_ar: 查询的日期，格式为 "YYYYMMDD"
    :param NOAAID: 活动区的编号
    :return: 如果is60为'T'且is120为'120'，则返回True，否则返回False
    """
    connection = None
    try:
        # 建立数据库连接
        connection = pymysql.connect(**db_config)
        cursor = connection.cursor()

        # 构建查询SQL语句
        query_sql = f"""
            SELECT is60, is120 FROM ar_flare_prediction
            WHERE AR_info_time = "{date_format_ar}" AND Nmbr = "{NOAAID}"
        """

        # 执行查询
        cursor.execute(query_sql)
        result = cursor.fetchone()  # 获取查询结果

        # 判断查询结果
        if result:
            is60, is120 = result
            return is60 == 'T' and is120 == '120'
        else:
            return False
    except pymysql.MySQLError as e:

        return False
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
if __name__ == '__main__':
    test_db_connection()
    print(check_is60_is120(date_format_ar=datetime.datetime.today().strftime("%Y%m%d") ,NOAAID=14050))