import requests
import re
import json
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


class FundNameCrawler:
    def __init__(self, fund_code: str):
        """
        :param fund_code: 基金代码
        """
        self.fund_code = fund_code
        self.session = requests.session()

    @staticmethod
    def content_formatter(content: str):
        """去掉jsonp外壳，转成dict"""
        params = re.compile(r'jQuery.+?\((.*)\)')
        data = json.loads(params.findall(content)[0])
        return data

    def get_name(self) -> str:
        """
        根据基金代码获取基金名称
        """
        search_url = 'https://fundsuggest.eastmoney.com/FundSearch/api/FundSearchAPI.ashx'
        params = {
            "callback": "jQuery18309325043269513131_1618730779404",
            "m": 1,
            "key": self.fund_code,
        }
        try:
            res = self.session.get(search_url, params=params, timeout=5)
            content = self.content_formatter(res.text)
            name = content['Datas'][0]['NAME']
            return name
        except Exception as e:
            print(f"基金 {self.fund_code} 获取名称失败: {e}")
            return ""


def run_one_fund(fund_code: str):
    crawler = FundNameCrawler(fund_code)
    name = crawler.get_name()
    time.sleep(0.2)  # 防止请求太快
    return fund_code, name


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

    results = {}

    # 启动 4 个进程并行获取
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_one_fund, code) for code in set(fund_list)]
        for future in as_completed(futures):
            try:
                code, name = future.result()
                if name:
                    print(f"{code} -> {name}")
                    results[code] = name
            except Exception as e:
                print(f"出错: {e}")

    # 保存为 JSON
    out_file = "fund_names.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n基金名称已保存到 {out_file}")
