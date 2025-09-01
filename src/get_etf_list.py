import re, requests, json
data = open('data-china/etf/etf.html', 'r', encoding='utf-8').read()

match = re.findall('data-href="(.*)" data-code="(.*)" data-name="(.*)">', data)

result = dict()
for i in match:
    result[i[1]] = {
        'code': i[1],
        'name': i[2],
        'link': 'https://zhishubao.1234567.com.cn/' + i[0]
    }

for i in result.values():
    data = requests.get(i['link']).text
    i['fund'] = re.findall('<p class="code">(.*?)</p>', data)
    with open('data-china/etf/etf.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(result, indent=4, ensure_ascii=False))