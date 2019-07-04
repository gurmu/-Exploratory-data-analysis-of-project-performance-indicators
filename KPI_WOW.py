# Load all packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from matplotlib import colors
from matplotlib import pyplot
from bokeh.io import output_notebook
from bokeh.charts import TimeSeries, show,Scatter
from bokeh.models import BoxSelectTool

output_notebook()

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# Preprocessing

create dataframe table
ww=pd.read_csv("./warcraft-avatar-history/wowah_data.csv")

# Inspect
ww.head()

# Rename columns to remove extra space
ww.rename(columns={' level':'level',' race':'race',' charclass': 'charclass', ' zone':'zone',' guild':'guild',' timestamp':'timestamp'}, inplace=True)

# Summarize data types of columns 
ww.info()

# Take a subsample of users and their associated values from the original dataframe to reduce processing 
# time of data analysis.
# Note: The following analysis will produce different results from similar analyses on the entire dataframe, especially
# for very small subsamples.
unique_char=pd.Series(ww.char.unique())
sample_unique_char=list(unique_char.sample(frac=0.03))
ww_char_sub=ww[ww['char'].isin(sample_unique_char)]


# Convert timestamp and date to datetime format and generate columns for date, month, day, hour
ww_char_sub.loc[:,'timestamp']=pd.to_datetime(ww_char_sub['timestamp'])

ww_char_sub.loc[:,'date']=ww_char_sub['timestamp'].dt.date
ww_char_sub.loc[:,'month_int']=ww_char_sub['timestamp'].dt.month
ww_char_sub.loc[:,'day']=ww_char_sub['timestamp'].dt.day
ww_char_sub.loc[:,'hour']=ww_char_sub['timestamp'].dt.hour

m={1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
ww_char_sub.loc[:,'month']=ww_char_sub['month_int'].map(m)

ww_char_sub=ww_char_sub.sort_values('timestamp', ascending=True)

ww_char_sub.loc[:,'date']=pd.to_datetime(ww_char_sub['date'])


# Inspect
ww_char_sub.head()


# Inpsect 
print ww_char_sub.shape


# Include the date that a user likely downloaded the game
add=ww_char_sub.groupby('char').agg({'timestamp':np.min}).reset_index()
add=add.rename(columns={'timestamp':'beg_date'})
ww_char_sub=ww_char_sub.merge(add, how="left", on="char")
print ww_char_sub.shape


# Include the days since install (dsi) as a column and convert install date to datetime format
ww_char_sub['dsi']=pd.Series([i.days for i in (ww_char_sub.timestamp-ww_char_sub.beg_date)])
ww_char_sub.loc[:,'beg_date']=ww_char_sub['beg_date'].dt.date


# Inspect
ww_char_sub.head()


# Inspect
ww_char_sub.shape


# Break levels into ranges
ww_char_sub['level_range']=pd.cut(ww_char_sub['level'],[0,60,70,80])


# Count unique users by date
dau=ww_char_sub.groupby(['date',])['char'].nunique().reset_index()


# Inpsect
dau.head()


# Set date as index
dau.set_index('date',inplace=True)


# Plot DAU
colors=sns.color_palette("Set2", 10)

fig, ax = plt.subplots(1,1, figsize=(20,15))

dau.loc[:,['char']].plot(color=colors[0],ax=ax,legend=False)

ax.set_ylabel("DAU Count",style='italic',fontsize=14)
ax.set_xlabel("Date",style='italic',fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)

pyplot.show()


# Count unique users by date and range of levels
dau_level=ww_char_sub.groupby(['date','level_range',])['char'].nunique().reset_index()


# Add a column to the dau_level dataframe that includes the sum of all unique users by date
print dau_level.shape
add=dau_level.groupby('date')['char'].agg({'char':np.sum}).reset_index()
add=add.rename(columns={'char':'percent_dau'})
dau_level=dau_level.merge(add,how='left',on="date")
print dau_level.shape


# Inspect
dau_level.head()


# Calculate the percent of DAU out of all DAU for the day
dau_level.loc[:,'percent_dau']=(dau_level['char']*100.0)/dau_level['percent_dau']

dau_level=dau_level.rename(columns={'char':'dau'})


# Inspect
dau_level.head()


# Plot DAU 
# Note: DAU percents that were similar over time were grouped. The grouping resulted in three distinct groups. 
fig, ax = plt.subplots(1,1, figsize=(20,15))


levels=list(dau_level['level_range'].unique())

for i,item in enumerate(levels):
    temp=dau_level[dau_level['level_range']==item]
    temp.set_index('date',inplace=True)
    temp.loc[:,['dau']].plot(color=colors[i],ax=ax)

ax.legend(levels,loc="best",fontsize='x-large')


# # plt.setp(labels, rotation=90) 
ax.set_ylabel("DAU Count",style='italic',fontsize=14)
ax.set_xlabel("Date",style='italic',fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)

pyplot.show()



# Plot DAU percent
fig, ax = plt.subplots(1,1, figsize=(20,15))


levels=list(dau_level['level_range'].unique())

for i,item in enumerate(levels):
    temp=dau_level[dau_level['level_range']==item]
    temp.set_index('date',inplace=True)
    temp.loc[:,['percent_dau']].plot(color=colors[i],ax=ax)

ax.legend(levels,loc="best",fontsize='x-large')


# # plt.setp(labels, rotation=90) 
ax.set_ylabel("DAU Percent (%)",style='italic',fontsize=14)
ax.set_xlabel("Date",style='italic',fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)

pyplot.show()



# Count unique users by date and range of levels
sess_dau_level=ww_char_sub.groupby(['date','level_range',])['char'].nunique().reset_index()
sess_dau_level=sess_dau_level.rename(columns={'char':'dau'})
sess_dau_level.head()


# Count all users by date and range of levels
temp=ww_char_sub.groupby(['date','level_range',])['char'].count().reset_index()
temp=temp.rename(columns={'char':'sessions'})
temp.head()


# Add column of count of all users by date and range of levels to the dataframe with the count of unique users by date
# and range of levels
sess_dau_level['level_date']=sess_dau_level['level_range'].astype(str).str.cat(sess_dau_level['date'].astype(str),sep=',')
temp['level_date']=temp['level_range'].astype(str).str.cat(temp['date'].astype(str),sep=',')
temp=temp.loc[:,['level_date','sessions']]

print sess_dau_level.shape
sess_dau_level=sess_dau_level.merge(temp,how='left',on="level_date")
print sess_dau_level.shape



# Calculate sessions per DAU
sess_dau_level.loc[:,'sessionsperdau']=(sess_dau_level['sessions']*1.0)/sess_dau_level['dau']


# Inspect
sess_dau_level.head()


# Plot sessions/DAU
temp=pd.pivot_table(sess_dau_level, values='sessionsperdau', index=['date'], columns=['level_range']).reset_index()

TOOLS="pan,wheel_zoom,box_zoom,reset,resize,save,crosshair,box_select"

p=TimeSeries(data=temp,x='date',y=['(0, 60]','(60, 70]','(70, 80]'], ylabel='Sessions per DAU',xlabel='Date',legend=True,plot_width=900, plot_height=700,tools=TOOLS,color=['green','orange','blue'])

show(p)


# Count unique users by month and range of levels
mau_level=ww_char_sub.groupby(['month_int','level_range',])['char'].nunique().reset_index()


# Add a column to the mau_level dataframe that includes the sum of all unique users by month
print mau_level.shape
add=mau_level.groupby('month_int')['char'].agg({'char':np.sum}).reset_index()
add=add.rename(columns={'char':'percent_mau'})
mau_level=mau_level.merge(add,how='left',on="month_int")
print mau_level.shape


# Inspect
mau_level.head()

# Calculate the percent of MAU out of all MAU for the month
mau_level.loc[:,'percent_mau']=(mau_level['char']*100.0)/mau_level['percent_mau']
mau_level=mau_level.rename(columns={'char':'mau'})


# Inspect
mau_level.head()


# Plot MAU by range of levels
temp=pd.pivot_table(mau_level, values='mau', index=['month_int'], columns=['level_range']).reset_index()

TOOLS="pan,wheel_zoom,box_zoom,reset,resize,save,crosshair,box_select"

p=TimeSeries(data=temp,x='month_int',y=['(0, 60]','(60, 70]','(70, 80]'], ylabel='MAU Count',xlabel='Month',legend=True,plot_width=900, plot_height=700,tools=TOOLS,color=['green','orange','blue'])

show(p)


# Plot MAU percent by range of levels
temp=pd.pivot_table(mau_level, values='percent_mau', index=['month_int'], columns=['level_range']).reset_index()

TOOLS="pan,wheel_zoom,box_zoom,reset,resize,save,crosshair,box_select"

p=TimeSeries(data=temp,x='month_int',y=['(0, 60]','(60, 70]','(70, 80]'], ylabel='MAU Percent (%)',xlabel='Month',legend=True,plot_width=900, plot_height=700,tools=TOOLS,color=['green','orange','blue'])

show(p)


# Add the MAU column to the dataframe including the DAU grouped by range of levels
mau_level_temp=mau_level.loc[:,['month_int','level_range','mau']]

mau_level_temp['level_month']=mau_level_temp['level_range'].astype(str).str.cat(mau_level_temp['month_int'].astype(str),sep=',')

mau_level_temp=mau_level_temp.loc[:,['level_month','mau']]

dau_level.loc[:,'month_int']=dau_level['date'].dt.month
dau_level_temp=dau_level.loc[:,['date','level_range','dau','month_int']]

dau_level_temp['level_month']=dau_level_temp['level_range'].astype(str).str.cat(dau_level_temp['month_int'].astype(str),sep=',')

print dau_level_temp.shape
dau_level_temp=dau_level_temp.merge(mau_level_temp,how='left',on="level_month")
print dau_level_temp.shape


# Inspect
dau_level_temp.head()


# Calculate percent of DAU/MAU
dau_level_temp.loc[:,'percent_dau_mau']=(dau_level_temp['dau']*100.0)/dau_level_temp['mau']

# Inspect
dau_level_temp.head()


# Plot DAU/MAU by range of levels
temp=pd.pivot_table(dau_level_temp, values='percent_dau_mau', index=['date'], columns=['level_range']).reset_index()

TOOLS="pan,wheel_zoom,box_zoom,reset,resize,save,crosshair,box_select"

p=TimeSeries(data=temp,x='date',y=['(0, 60]','(60, 70]','(70, 80]'], ylabel='DAU/MAU Percent (%)',xlabel='Date',legend=True,plot_width=900, plot_height=700,tools=TOOLS,color=['green','orange','blue'])

show(p)

# Count unique users by install date and DSI
retention_group=ww_char_sub.groupby(['beg_date','dsi',])['char'].nunique().reset_index()

retention_group=retention_group.rename(columns={'char':'yes'})

# Inspect
retention_group.head()



# Add to retention dataframe count of unique users when dsi is equal to 0 
no=retention_group.loc[retention_group['dsi']==0,['beg_date','yes']]
no=no.rename(columns={'yes':'no'})

print retention_group.shape
retention_group=retention_group.merge(no, how="left", on="beg_date")
print retention_group.shape


# For all user with the same beginning date, calculate the difference of the count of unique users when the dsi is 
# equal to 0 and count of unique users when dsi is equal to or greater than 0
retention_group.loc[:,'no']=retention_group['no']-retention_group['yes']


# For each of the unique beginning dates, calculate the size of retained and churned users
retention_group.loc[:,'size']=retention_group['no']+retention_group['yes']


# Inspect
retention_group.head()


# Calculate the sum of all retained and churned users by dsi
retention=retention_group.groupby(['dsi'])['yes','no'].sum().reset_index()


# Inspect
retention.head()


# Caculate the sum of the retained and churned users for dsi>0 and calculate percent retained
retention.loc[:,'size']=retention['yes']+retention['no']
retention.loc[:,'percent_ret']=(retention['yes']*100.0)/retention['size']
retention=retention.loc[retention['dsi']>0,:]


# Inspect
retention.head()


# Plot retention count over time
temp=retention.loc[:,['dsi','yes']]

TOOLS="pan,wheel_zoom,box_zoom,reset,resize,save,crosshair,box_select"

p=TimeSeries(data=temp,x='dsi',y=['yes'], ylabel='Retention Count',xlabel='DSI',legend=False,plot_width=900, plot_height=700,tools=TOOLS,color='green')

show(p)


# Plot retention percent over time
temp=retention.loc[:,['dsi','percent_ret']]

TOOLS="pan,wheel_zoom,box_zoom,reset,resize,save,crosshair,box_select"

p=TimeSeries(data=temp,x='dsi',y=['percent_ret'], ylabel='Retention Percent (%)',xlabel='DSI',legend=False,plot_width=900, plot_height=700,tools=TOOLS,color='green')

show(p)

# Count unique users by range of levels, beginning date, and dsi
retention_group_level=ww_char_sub.groupby(['level_range','beg_date','dsi',])['char'].nunique().reset_index()
retention_group_level=retention_group_level.rename(columns={'char':'yes'})


# Inpsect
retention_group_level.head()


# Add to retention dataframe count of unique users when dsi is equal to 0 
retention_group_level.loc[:,'level_date']=retention_group_level['level_range'].astype(str).str.cat(retention_group_level['beg_date'].astype(str),sep=',')

no=retention_group_level.loc[retention_group_level['dsi']==0,['level_date','yes']]
no=no.rename(columns={'yes':'no'})

print retention_group_level.shape
retention_group_level=retention_group_level.merge(no, how="left", on="level_date")
print retention_group_level.shape


# Inspect
retention_group_level.head()


# For all user with the same beginning date, calculate the difference of the count of unique users when the dsi is 
# equal to 0 and count of unique users when dsi is equal to or greater than 0
# For each of the unique beginning dates, calculate the size of retained and churned users
retention_group_level.loc[:,'no']=retention_group_level['no']-retention_group_level['yes']
retention_group_level.loc[:,'size']=retention_group_level['no']+retention_group_level['yes']


# Inspect
retention_group_level.head()


# Calculate the sum of all retained and churned users by range of levels and dsi
retention_level=retention_group_level.groupby(['level_range','dsi'])['yes','no'].sum().reset_index()


# For each range of levels, caculate the sum of the retained and churned users for dsi>0 and calculate percent retained
retention_level.loc[:,'size']=retention_level['yes']+retention_level['no']
retention_level.loc[:,'percent_ret']=(retention_level['yes']*100.0)/retention_level['size']
retention_level=retention_level.loc[retention_level['dsi']>0,:]


# Plot retention count over time for each range of levels
temp=pd.pivot_table(retention_level, values='yes', index=['dsi'], columns=['level_range']).reset_index()

TOOLS="pan,wheel_zoom,box_zoom,reset,resize,save,crosshair,box_select"

p=TimeSeries(data=temp,x='dsi',y=['(0, 60]','(60, 70]','(70, 80]'], ylabel='Retention Count',xlabel='DSI',legend=True,plot_width=900, plot_height=700,tools=TOOLS,color=['green','orange','blue'])

show(p)



# Plot retention percent over time for each range of levels
temp=pd.pivot_table(retention_level, values='percent_ret', index=['dsi'], columns=['level_range']).reset_index()

TOOLS="pan,wheel_zoom,box_zoom,reset,resize,save,crosshair,box_select"

p=TimeSeries(data=temp,x='dsi',y=['(0, 60]','(60, 70]','(70, 80]'], ylabel='Retention Percent (%)',xlabel='DSI',legend=True,plot_width=900, plot_height=700,tools=TOOLS,color=['green','orange','blue'])

show(p)


# Count unique users by level
levels_uplayers=ww_char_sub.groupby(['level'])['char'].nunique().reset_index()


# Inspect
levels_uplayers.head()


# Plot the number of unique users who have compledted each level vs. level
TOOLS="pan,wheel_zoom,box_zoom,reset,resize,save,crosshair,box_select"
p=Scatter(levels_uplayers,x='level',y='char',xlabel="Level",ylabel="Unique Users",tools=TOOLS)
show(p)


