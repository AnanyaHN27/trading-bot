#!/usr/bin/env python
# coding: utf-8

# # Investigating:
#    ### Whether returns can be described with a normal distribution, if there is directional bias in daily change and if price movement can be described as a random walk.

# In[3]:


pip install pandas_datareader


# In[5]:


pip install seaborn


# In[17]:


pip install yfinance


# In[57]:


import numpy as np
import pandas as pd
import pandas_datareader as pdr #to access live data from yahoo api
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 8, 6
import seaborn as sns
sns.set_theme()


# In[58]:


import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)
from pandas_datareader import data as pdr

amzn = pdr.get_data_yahoo('AMZN').reset_index()


# In[59]:


#we can see that we have now got the ohlcv data for historical data from 1997 onwards
amzn


# In[60]:


amzn.columns


# In[61]:


plottable = amzn[amzn['Date'] > '2020-01-01']

plt.plot(plottable['Date'], plottable['Close'])


# In[62]:


#store the instantaneous rate of return in a separate series
amzn_close = amzn[amzn['Date'] > '2020-01-01']['Close']
amzn_dates = amzn[amzn['Date'] > '2020-01-01']['Date']
amzn_return = np.log(amzn_close).diff()
amzn_return.head()


# In[63]:


amzn_return.dropna(inplace=True)


# In[64]:


amzn_return


# In[46]:


plt.plot(amzn_dates, amzn_close)


# In[39]:


from scipy import stats


# In[54]:



n, minmax, mean, var, skew, kurt = stats.describe(amzn_return)
mini, maxi = minmax
std = var ** 0.5 #getting the standard deviation
print(n, minmax, mean, var, skew, kurt)


# In[52]:


from scipy.stats import norm


# In[53]:


x = norm.rvs(mean, std, n)
stats.describe(x) #these are different which is kinda wild


# Plot histogram of price changes with normal curve overlay

# In[55]:


plt.hist(amzn_return, bins = 25, edgecolor='w', density= True)
data = np.linspace(mini, maxi, 100)
plt.plot(data, norm.pdf(data, mean, std));


# In[56]:





# In[ ]:




