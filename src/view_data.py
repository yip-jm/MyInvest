import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

count = 1
fig = 1
fund_info = dict()

for fn in os.listdir('data-china'):
    df = pd.read_csv(os.path.join('data-china', fn))
    df = df[df['FSRQ'] >= '2019-00-00']
    df['log_ret'] = np.log(pd.to_numeric(df['LJJZ']))
    k, b, r, p, err = ss.linregress(range(len(df), 0, -1), df['log_ret'].values)
    fund_info[fn] = {
        'k': k * 10000,
        'b': b,
        'r': r,
        'p': p,
        'err': err
    }

fund_list = os.listdir('data-china')
fund_list.sort(key=lambda x: round(fund_info[x]['k'] / (1 - fund_info[x]['r']), 2), reverse=True)
# print(list(map(lambda x: fund_info[x]['r'], fund_list)))
# exit()
# print(fund_list)

for fn in fund_list:
    try:
        plt.subplot(5, 5, count)
        count += 1
        df = pd.read_csv(os.path.join('data-china', fn))
        df = df[df['FSRQ'] >= '2019-00-00']
        plt.plot(pd.to_datetime(df['FSRQ']), df['LJJZ'])
        plt.title(f'{fn}\nk={round(fund_info[fn]["k"], 2)}\nr={round(fund_info[fn]["r"], 2)}')
        if count > 25:
            plt.show()
            count = 1
            fig += 1
    except:
        print(fn)

plt.show()
        