# -*- coding: utf-8 -*-
import matplotlib
# matplotlib.use('TkAgg')

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA

# %matplotlib inline

# init params
datafile = '../Datasets/arima_data.csv'
forecastnum = 5  # 预测的天数

# 读取数据，指定日期列为指标，Pandas自动将"DATE"列识别为Datetime格式
data = pd.read_csv(datafile, index_col='DATE')
# data[:5]

# 时序图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
data.plot()
plt.show()

# 自相关图
# plot_acf(data).show()
plot_acf(data)

# 平稳性检测
salesVolumnADF = ADF(data['SALES_VOLUME'])
print(u'Raw time series ADF test result：\n', salesVolumnADF)
# 返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore

# 差分后的结果
D_data = data.diff().dropna()
D_data.columns = [u'SALES_VOLUME_DIFF']
D_data.plot()  # 时序图
plot_acf(D_data)  # 自相关图
plot_pacf(D_data)  # 偏自相关图
plt.show()

salesVolumnDiffADF = ADF(D_data['SALES_VOLUME_DIFF'])
print(u'1-diff time series ADF test result：', salesVolumnDiffADF)  # 平稳性检测

print("Sales Volumn ADF", salesVolumnADF)
print("Sales Volumn Diff ADF", salesVolumnDiffADF)
# 返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore

# 差分序列的白噪声检验
print(u'1-diff series white noise test result: ', acorr_ljungbox(D_data, lags=1))  # 返回统计量和p值

data[u'SALES_VOLUME'] = data[u'SALES_VOLUME'].astype(float)
# 定阶
pmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
qmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
bic_matrix = []  # bic矩阵
for p in range(pmax + 1):
    tmp = []
    for q in range(qmax + 1):
        try:  # 存在部分报错，所以用try来跳过报错。
            tmp.append(ARIMA(data, (p, 1, q)).fit().bic)
        except:
            tmp.append(None)
    bic_matrix.append(tmp)

bic_matrix = pd.DataFrame(bic_matrix)  # 从中可以找出最小值

p, q = bic_matrix.stack().idxmin()  # 先用stack展平，然后用idxmin找出最小值位置。
print(u'BIC minimal p-value and q-value is：%s、%s' % (p, q))
model = ARIMA(data, (p, 1, q)).fit()  # 建立ARIMA(0, 1, 1)模型

# 给出一份模型报告
print("************************************************************")
print(model.summary2())
print("************************************************************")
print()
print("************************************************************")
print(model.forecast(forecastnum))  # 作为期forecastnum天的预测，返回预测结果、标准误差、置信区间。
