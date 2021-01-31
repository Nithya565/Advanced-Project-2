#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv (r'Cross-Border 2019.csv')


# In[3]:


df.head()


# In[5]:


df.tail()


# In[4]:


df.describe()


# In[5]:


new_df= df["Time (CET)"].str.split("-",n=1,expand= True)
df["Date part "]=new_df[0]
df["CET_Timestamp"]=new_df[1]


# In[6]:


df['CET_Timestamp'] = df['CET_Timestamp'].astype('datetime64[ns]')


# In[7]:


df = df.assign(Date= df.CET_Timestamp.dt.date, Time = df.CET_Timestamp.dt.time )
df = df.set_index (['CET_Timestamp'])


# In[8]:



df.head(3)


# In[9]:


df.index


# In[10]:


# Add columns with year, month, and weekday name
df['Year'] = df.index.year
df['Month'] = df.index.month
df['day'] = df.index.day
# Display a random sampling of 5 rows
# df.sample(5, random_state=0)


# In[11]:


df.tail(3)


# In[12]:


df.loc['2019-10-25']


# In[13]:


import matplotlib.pyplot as plt


# In[14]:


import seaborn as sns
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(11, 4)})


# In[15]:


cols_plot = ['BZN|FR > BZN|DE-LU [MW]','BZN|DE-LU > BZN|FR [MW]' ]
axes = df[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('cross border physical flows')


# In[16]:


ax = df.loc['2019', 'BZN|FR > BZN|DE-LU [MW]'].plot(grid = True)
ax.set_ylabel('cross border physical flows');


# In[17]:


ax = df.loc['2019', 'BZN|DE-LU > BZN|FR [MW]'].plot(grid = True)
ax.set_ylabel('cross border physical flows');


# In[18]:


# Specify the data columns we want to include (i.e. exclude Year, Month, Weekday Name)
data_columns = ['BZN|FR > BZN|DE-LU [MW]', 'BZN|DE-LU > BZN|FR [MW]']
# Resample to weekly frequency, aggregating with mean
df_mean = df[data_columns].resample('W').mean()
df_mean.head(3)


# In[19]:


# Start and end of the date range to extract
start, end = '2019-01-19', '2019-01-20'
# Plot daily and weekly resampled time series together
fig, ax = plt.subplots()
ax.plot(df.loc[start:end, 'BZN|FR > BZN|DE-LU [MW]'],
marker='.', linestyle='-', linewidth=0.5, label='Daily')
ax.plot(df_mean.loc[start:end, 'BZN|FR > BZN|DE-LU [MW]'],
marker='o', markersize=8, linestyle='-', label='Weekly Mean Resample')
ax.set_ylabel('cross border physical flows')
ax.legend();


# In[20]:


# Start and end of the date range to extract
start, end = '2019-01-19', '2019-01-20'
# Plot daily and weekly resampled time series together
fig, ax = plt.subplots()
ax.plot(df.loc[start:end, 'BZN|DE-LU > BZN|FR [MW]'],
marker='.', linestyle='-', linewidth=0.5, label='Daily')
ax.plot(df_mean.loc[start:end, 'BZN|DE-LU > BZN|FR [MW]'],
marker='o', markersize=8, linestyle='-', label='Weekly Mean Resample')
ax.set_ylabel('cross border physical flows')
ax.legend();


# In[21]:


df = np.random.rand(4, 6)


# In[22]:


df = np.random.rand(4, 6)
heat_map = sns.heatmap(df, annot=True)
  


# In[16]:


plt.rcParams['figure.figsize'] = (8, 6) # change plot size
df['BZN|FR > BZN|DE-LU [MW]'].resample('A').max().plot(kind='bar')
plt.title('Yearly Maximum cross border flows frequency for year end in BZN|FR > BZN|DE-LU [MW] ')


# In[40]:


plt.rcParams['figure.figsize'] = (16, 6) # change plot size
df['BZN|FR > BZN|DE-LU [MW]'].resample('W').max().plot(kind='bar')
plt.title(' Maximum cross border flows for week  in BZN|FR > BZN|DE-LU [MW] ')


# In[20]:


plt.rcParams['figure.figsize'] = (16, 6) # change plot size
df['BZN|FR > BZN|DE-LU [MW]'].resample('M').max().plot(kind='bar')
plt.title('Yearly Maximum cross border flows for month end in BZN|FR > BZN|DE-LU [MW] ')


# In[21]:


plt.rcParams['figure.figsize'] = (16, 6) # change plot size
df['BZN|FR > BZN|DE-LU [MW]'].resample('Q').max().plot(kind='bar')
plt.title('Yearly Maximum cross border flows for quarter end  in BZN|FR > BZN|DE-LU [MW] ')


# In[17]:


plt.rcParams['figure.figsize'] = (8, 6) # change plot size
df['BZN|DE-LU > BZN|FR [MW]'].resample('A').max().plot(kind='bar')
plt.title('Yearly Maximum cross border flows frequency for year end in BZN|DE-LU > BZN|FR [MW]')


# In[18]:


plt.rcParams['figure.figsize'] = (16, 6) # change plot size
df['BZN|DE-LU > BZN|FR [MW]'].resample('W').max().plot(kind='bar')
plt.title('Yearly Maximum cross border flows for week end in BZN|DE-LU > BZN|FR [MW] ')


# In[19]:


plt.rcParams['figure.figsize'] = (16, 6) # change plot size
df['BZN|DE-LU > BZN|FR [MW]'].resample('M').max().plot(kind='bar')
plt.title('Yearly Maximum cross border flows for month end in BZN|DE-LU > BZN|FR [MW] ')


# In[22]:


plt.rcParams['figure.figsize'] = (16, 6) # change plot size
df['BZN|DE-LU > BZN|FR [MW]'].resample('Q').max().plot(kind='bar')
plt.title('Yearly Maximum cross border flows for quarter end  in BZN|DE-LU > BZN|FR [MW] ')


# In[160]:


sales_by_day = df.groupby('day').size()
plot_by_day = sales_by_day.plot(title='Daily cross border flows',xticks=(range(1,31)),rot=55)
plot_by_day.set_xlabel('Day')
plot_by_day.set_ylabel('BZN|DE-LU > BZN|FR [MW]')
plt.title("Daily cross border flows analysis")


# In[170]:


sales_by_day = df.groupby('Month').size()
plot_by_day = sales_by_day.plot(title='Daily cross border flows',xticks=(range(1,31)),rot=55)
plot_by_day.set_xlabel('Month')
plot_by_day.set_ylabel('BZN|FR > BZN|DE-LU [MW]	')
plt.title("Monthly cross border flows analysis")


# In[5]:


corr = df.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df.columns)
ax.set_yticklabels(df.columns)
plt.show()


# In[37]:


pt_titanic = df.pivot_table(index='Time', columns='Date', values='BZN|FR > BZN|DE-LU [MW]')


# In[38]:



plt.figure(figsize=(16,9))
sns.heatmap(pt_titanic,cmap="coolwarm")


# In[51]:


pt_titanic2 = df.pivot_table(index='Time', columns='Date', values='BZN|DE-LU > BZN|FR [MW]')


# In[53]:


plt.figure(figsize=(16,9))
sns.heatmap(pt_titanic2,cmap="coolwarm")


# In[63]:


pt_titanic2 = df.pivot_table(index='Time', columns='Month', values='BZN|DE-LU > BZN|FR [MW]')


# In[64]:


plt.figure(figsize=(16,9))
sns.heatmap(pt_titanic2,cmap="coolwarm")


# In[68]:


pt_titanic3 = df.pivot_table(index='Time', columns='Month', values='BZN|FR > BZN|DE-LU [MW]')


# In[69]:


plt.figure(figsize=(16,9))
sns.heatmap(pt_titanic3,cmap="coolwarm")


# In[88]:


pt_titanic4 = df.pivot_table(index='Time', columns='day', values='BZN|FR > BZN|DE-LU [MW]')


# In[89]:


plt.figure(figsize=(16,9))
sns.heatmap(pt_titanic4,cmap="coolwarm")


# In[90]:


pt_titanic5 = df.pivot_table(index='Time', columns='day', values='BZN|DE-LU > BZN|FR [MW]')


# In[91]:


plt.figure(figsize=(16,9))
sns.heatmap(pt_titanic5,cmap="coolwarm")


# In[98]:


pt_titanic6 = df.pivot_table(index='Month', columns='day', values='BZN|DE-LU > BZN|FR [MW]')


# In[99]:


plt.figure(figsize=(16,9))
sns.heatmap(pt_titanic6,cmap="coolwarm")


# In[100]:


pt_titanic7 = df.pivot_table(index='Month', columns='day', values='BZN|FR > BZN|DE-LU [MW]')


# In[101]:


plt.figure(figsize=(16,9))
sns.heatmap(pt_titanic7,cmap="coolwarm")


# In[113]:





# In[ ]:




