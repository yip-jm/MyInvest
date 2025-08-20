
# -*- coding: utf-8 -*-
# @Author  : AwesomeTang
# @File    : crawler.py
# @Version : Python 3.7
# @Time    : 2021-04-18 12:17

import requests
import re
import json
import pandas as pd
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed



class FundCrawler:
    def __init__(self,
                 fund_code: int,
                 page_range: int = None,
                 file_name=None):
        """
        :param fund_code:  基金代码
        :param page_range:  获取最大页码数，每页包含20天的数据
        """
        self.root_url = 'http://api.fund.eastmoney.com/f10/lsjz'
        self.fund_code = fund_code
        self.session = requests.session()
        self.page_range = page_range
        self.file_name = file_name if file_name else '{}.csv'.format(self.fund_code)
        self.headers = {
            'Host': 'api.fund.eastmoney.com',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
            'Referer': 'http://fundf10.eastmoney.com/jjjz_%s.html' % self.fund_code,
        }

    def fount_info(self):
        search_url = 'https://fundsuggest.eastmoney.com/FundSearch/api/FundSearchAPI.ashx'
        params = {
            "callback": "jQuery18309325043269513131_1618730779404",
            "m": 1,
            "key": self.fund_code,
        }
        res = self.session.get(search_url,
                               params=params
                               )
        try:
            content = self.content_formatter(res.text)
            fields = '，'.join([i['TTYPENAME'] for i in content['Datas'][0]['ZTJJInfo']])
            print("{:*^30}".format(self.fund_code))
            print("* 基金名称：{0:{1}>10} *".format(content['Datas'][0]['NAME'], chr(12288)))
            print("* 基金类型：{0:{1}>10} *".format(content['Datas'][0]['FundBaseInfo']['FTYPE'], chr(12288)))
            print("* 基金经理：{0:{1}>10} *".format(content['Datas'][0]['FundBaseInfo']['JJJL'], chr(12288)))
            print("* 相关领域：{0:{1}>10} *".format(fields, chr(12288)))
            print("*" * 30)
            return True
        except TypeError:
            print('Fail to pinpoint fund code, {}, please check.'.format(self.fund_code))

    @staticmethod
    def content_formatter(content):
        params = re.compile('jQuery.+?\((.*)\)')
        data = json.loads(params.findall(content)[0])
        return data

    def page_data(self,
                  page_index):
        params = {
            'callback': 'jQuery18308909743577296265_1618718938738',
            'fundCode': self.fund_code,
            'pageIndex': page_index,
            'pageSize': 20,
        }
        res = self.session.get(url=self.root_url, headers=self.headers, params=params)
        content = self.content_formatter(res.text)
        return content

    def page_iter(self):
        for page_index in range(self.page_range):
            item = self.page_data(page_index + 1)
            yield item

    def get_all(self):
        total_count = float('inf')
        page_index = 0
        while page_index * 20 <= total_count:
            item = self.page_data(page_index + 1)
            page_index += 1
            total_count = item['TotalCount']
            yield item

    def run(self):
        if self.fount_info():
            fn = 'data-china/' + self.file_name
            max_time = '0000-00-00'
            df = None
            if os.path.isfile(fn):
                df = pd.read_csv(fn)
                max_time = df.max()['FSRQ']
                print('已有数据，截至 {}'.format(max_time))
            else:
                df = pd.DataFrame()
            # 获取目前的数据到多少号了
            
            if self.page_range:
                for data in self.page_iter():
                    tmp = data['Data']['LSJZList']
                    min_time = min(tmp, key=lambda x: x['FSRQ'])
                    if min_time < max_time:
                        tmp = [i for i in tmp if i['FSRQ'] > max_time]
                    df = df.append(tmp, ignore_index=True)
                    if min_time < max_time:
                        break
            else:
                for data in self.get_all():
                    tmp = data['Data']['LSJZList']
                    min_time = min(tmp, key=lambda x: x['FSRQ'])['FSRQ']
                    if min_time < max_time:
                        tmp = [i for i in tmp if i['FSRQ'] > max_time]
                    tmp = pd.DataFrame(tmp)   # 转换成 DataFrame
                    df = pd.concat([df, tmp], ignore_index=True)    
                    if min_time < max_time:
                        break
                    print("\r{:*^30}".format(' DOWNLOADING '), end='')
            df = df.sort_values(by='FSRQ', ascending=False)
            df.to_csv('data-china/' + self.file_name)
            print("\r{:*^30}".format(' WORK DONE '))
            print("{:*^30}".format(' FILE NAME '))
            print("*{:^28}*".format(self.file_name))
            print("*" * 30)

def run_one_fund(fundCode):
    c = FundCrawler(fundCode)
    c.run()
    time.sleep(2)  # 防止接口压力太大
    return fundCode


    
if __name__ == "__main__":
    fund_list = [
        '165513', '001322', '002170', '004475', '000628', '007751',
        '161128', '210002', '001917', '008271', '006373', '090007',
        '090013', '000043', '000218', '002849', '000979', '008115',
        '260112', '005576', '160323', '004011', '007280', '515450',
        '000369', '003496', '006212', '007925', '007540', '009689',
        '001745', '001806', '270042', '050025', '486002', '501025',
        '320013', '000480', '002276', '006700', '110017', '513500',
        '166301', '006624', '007509', '004246', '420009', '001122',
        '217021', '210009', '004279', '007492', '001405', '005908',
        '002863', '162102', '539001', '540007', '750002', '009005',
        '006980', '519669', '003545', '000931', '004750', '360008',
        '002236', '004206', '001167', '002833', '008269', '005984',
        '006961', '007107', '005443', '000573', '005561', '512890',
        '007466', '002049', '510080', '511270', '003376',
        '003175', '003591', '003102', '007676'
    ]

    # 启动 4 个进程
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_one_fund, code) for code in set(fund_list)]
        for future in as_completed(futures):
            try:
                code = future.result()
                print(f"基金 {code} 下载完成")
            except Exception as e:
                print(f"出错: {e}")