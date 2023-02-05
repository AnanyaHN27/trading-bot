#!/usr/bin/env python
# coding: utf-8

# In[146]:


import numpy as np
import pandas as pd
import pandas_datareader as pdr #to access live data from yahoo api
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = (20,10)
rcParams["keymap.pan"]
import seaborn as sns
sns.set_theme()


# In[34]:


import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)
from pandas_datareader import data as pdr

amzn = pdr.get_data_yahoo('AMZN').reset_index()


# In[49]:


plottable = amzn[amzn['Date'] > '2020-01-01']


# In[64]:


amzn_close = amzn[amzn['Date'] > '2020-01-01']['Close']
amzn_close_dates = amzn[amzn['Date'] > '2020-01-01']['Date']

amzn_close_dates.head()


# In[54]:


amzn_close.dropna(inplace=True)


# In[56]:


amzn_close


# In[55]:


amzn_close


# In[81]:


ema_series_20 = amzn_close.rolling(window=20).mean()
ema_series_20.fillna(0)
ema_series_50 = amzn_close.rolling(window=50).mean()
ema_series_50.fillna(0)


# In[82]:


whole_thing = pd.DataFrame({'Date': amzn_close_dates, '20 day': ema_series_20, '50 day': ema_series_50})


# In[84]:


whole_thing.fillna(0)


# In[152]:


ema_20_arr = ema_series_20.to_numpy()
ema_50_arr = ema_series_50.to_numpy()
date_arr = whole_thing['Date'].to_numpy()

idx = np.argwhere(np.diff(np.sign(ema_20_arr - ema_50_arr))).flatten()

BUY = []
buy_idx = []
SELL = []
sell_idx = []

amzn['MACD Decision'] = "WAIT"

for i, index_val in np.ndenumerate(idx):
    if ema_20_arr[index_val - 1] > ema_50_arr[index_val - 1]:
        amzn['MACD Decision'].iloc[index_val] = "SELL"
        SELL.append(index_val)
    else:
        amzn['MACD Decision'].iloc[index_val] = "BUY"
        BUY.append(index_val)

print(amzn['MACD Decision'].unique())


# In[133]:


#implement rsi

#BUY → MACD crossover with BUY signal & RSI in oversold region
#SELL → MACD crossover with SELL signal & RSI in overbought region

def getRS(series_obj):
    x = series_obj.to_numpy()
    pos = np.where(x > 0)
    neg = np.where(x <= 0)
    return np.sum(pos)/np.sum(neg)

amzn['Price Change'] = amzn['Open'] - amzn['Close']
amzn['Change Type'] = amzn['Price Change'].apply(lambda x: "Up" if x > 0 else "Down")
amzn['RS'] = amzn['Price Change'].rolling(window=14).apply(getRS)


# In[134]:


amzn['RS'][:16]


# In[135]:


amzn['RSI'] = 100/(1 + amzn['RS'])


# In[136]:


amzn['RSI']


# In[160]:


amzn['RSI Decision'] = amzn['RSI'].apply(lambda x: "BUY" if x < 30 else "SELL" if x > 70 else "WAIT")

conditions = [
    amzn['MACD Decision'].eq('BUY') & amzn['RSI Decision'].eq('BUY'),
    amzn['MACD Decision'].eq('SELL') & amzn['RSI Decision'].eq('SELL')
]

choices = ["BUY","SELL"]

amzn['MACD RSI Decision'] = np.select(conditions, choices, default="WAIT")


# In[177]:


idx_overall_buy = np.argwhere((amzn['MACD RSI Decision']=="BUY").to_numpy()).flatten()
idx_overall_sell = np.argwhere((amzn['MACD RSI Decision']=="SELL").to_numpy()).flatten()

close_nump = amzn['Close'].iloc[idx_overall_sell]

print(close_nump)

print(idx_overall_buy, idx_overall_sell)


# In[176]:


rsi_buy = pd.DataFrame({'Close': amzn[amzn['RSI Decision'] == "BUY"]['Close'], 'Date': amzn[amzn['RSI Decision'] == "BUY"]['Date']})
rsi_sell = pd.DataFrame({'Close': amzn[amzn['RSI Decision'] == "SELL"]['Close'], 'Date': amzn[amzn['RSI Decision'] == "SELL"]['Date']})

#plt.plot(rsi_buy['Date'], rsi_buy['Close'], color='black', label = "RSI buy regions")
#plt.plot(rsi_sell['Date'], rsi_sell['Close'], color='red', label = "RSI sell regions")

plt.title('MACD: crossover of 20 day ma and 50 ma for Amazon from 2020')
plt.plot(whole_thing['Date'], ema_20_arr, label = "20 day ma")
plt.plot(whole_thing['Date'], ema_50_arr, label = "50 day ma")

#plt.plot(whole_thing['Date'].to_numpy()[idx_overall_buy], amzn['Close'].to_numpy()[idx_overall_buy], '*', color = 'purple', label = "BUY")
#plt.plot(whole_thing['Date'].to_numpy()[idx_overall_sell], amzn['Close'].to_numpy()[idx_overall_sell], '*', color = 'green', label = "SELL")


plt.plot(date_arr[BUY], ema_20_arr[BUY], '*', color = 'purple', label="BUY")
plt.plot(date_arr[SELL], ema_20_arr[SELL], '*', color = 'green', label="SELL")

plt.legend()


# In[ ]:




