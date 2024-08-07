"""
Created on Fri Aug 2 10:59:59 2024

@author: bigett
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
'''This code computes the precipitation over a lake during the cold seasons using Pritchard's method (DOI: 10.1175/JHM-D-20-0206.1) on the lake water pressure series. 
The input data are the pressure series and a list of the dates of all the rainfall events that happened during the recording. The dates on the list were determined manually basing on the pressure series.
The estimated precipitation is supposed here to be equal to the difference of water column before and after the event taking account of the drainage rate  of the lake.
This drainage rate is supposed linear and is extrapolated here at the start and the end of each event with a least square fitting.'''


#flag for plotting the linear regressions
plot_event = True

#importing the dates of the events and the pressure series
path_golojang_b = r"golojang_b.csv"
path_dates = r"dates_events.csv"

df = pd.read_csv(path_golojang_b) 
dates_events = pd.read_csv(path_dates)

#formating the data
df['date'] = pd.to_datetime(df['date'])
dates_events['date_start'] = pd.to_datetime(dates_events['date_start'])
dates_events['date_end'] = pd.to_datetime(dates_events['date_end'])


total_precip = [] #will be the list with all the computed precipitation

# dates_events = dates_events[:] you can uncomment if you want to reduce the size of the number of events computed


def drainage_regression(event_id, position = 'start'):
    '''compute the linear regression on the drainage at the start or the end of an event'''
    #the regression is made on a time interval defined individually prior the running of the code. You can consult them on the file dates_envents.csv in the column 'timedelta_lr'
    
    #defining the dates of the start and the end of the regression
    if position == 'end':
        end_date = pd.to_datetime(end_date_event) + pd.to_timedelta(str(dates_events['timedelta_lr'].iloc[event_id])) 
        start_date = end_date_event

    else:        
        start_date = pd.to_datetime(start_date_event) - pd.to_timedelta(str(dates_events['timedelta_lr'].iloc[event_id])) 
        end_date = start_date_event
        
    
    # defining the time iterval of the regression
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df_interval = df.loc[mask]
    
    # Convert dates
    X = (df_interval['date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    X = X.values.reshape(-1, 1)
    y = df_interval['millimeters'].values
    
    # Performing the linear regression
    model = LinearRegression()
    model.fit(X, y)
    X_full = (df['date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    X_full = X_full.values.reshape(-1, 1)
    #returning the values of the computed line and the values of the pressure series during the time range of the regression
    return model.predict(X_full), df_interval




def mask_winter_pre_monsoon(df, start_year):
    '''will be used to compute the precipitation during the period DJFMAM of each year'''
    start_date = pd.Timestamp(f'{start_year}-12-01')
    end_date = pd.Timestamp(f'{start_year + 1}-05-31')
    
    mask = (df.index >= start_date) & (df.index <= end_date)
    return mask




#Now we can compute the precipitation for each event
for i in range(len(dates_events['date_start'])):
    
    # defines the dates of the beginnings and the ends of each event
    start_date_event = dates_events['date_start'].iloc[i]
    end_date_event = dates_events['date_end'].iloc[i]
    
    
    ##################### linear regression on the begining of the event and saving of the pressure series during the time range of the regression
    predictions_full_1, regression_interval_start = drainage_regression(i)
    
    ##################### linear regression on the end of the event and saving of the pressure series during the time range of the regression
    predictions_full_2, regression_interval_end = drainage_regression(i, 'end')
    
    ##################### finaly the water column equivalent of the event is computed using Pritchard's method
   
    #the total precipitation during an event is here supposed to be the distance between the start and the end regression at the end of the event 
    if pd.to_datetime(start_date_event) != pd.to_datetime('2022-04-09 02:00:00'): #using only the begining regression on the last event since the pressure series stop before the end of the event
        predictions_full = predictions_full_2 - predictions_full_1 #difference of the water column before and avter the event
        precip_one_event = predictions_full[df[df['date'] == end_date_event].index]
    
    else:       
        predictions_full = predictions_full_1 #difference of the water column before and avter the event
        precip_one_event = df[df['date'] == end_date_event]['millimeters'] - predictions_full[df[df['date'] == end_date_event].index]
        precip_one_event = [precip_one_event.iloc[0]]
    
    #adding the computed accumulation for the event to the list
    total_precip.append([start_date_event, end_date_event, precip_one_event[0]])
    
  
    if plot_event:
        #plot the regressions and the time intervals used for the regressions for each events if plot_event is set on True
        plt.figure(figsize=(12, 7))
        plt.plot(df['date'], df['millimeters'], label='original data', color='gray') #plot of the original pressure series
        
        plt.scatter(regression_interval_start['date'], regression_interval_start['millimeters'], color='blue', label='start interval') #coloring the pressure series in blue during the first regression time interval
        plt.scatter(regression_interval_end['date'], regression_interval_end['millimeters'], color='green', label='end interval') #coloring the pressure series in green during the second regression time interval

        plt.plot(df['date'], predictions_full_1, color='red', linewidth=2, label='linear regression') #plot the line of the first regression
        
        if pd.to_datetime(start_date_event) != pd.to_datetime('2022-04-09 02:00:00'):
            plt.plot(df['date'], predictions_full_2, color='red', linewidth=2, label='linear regression') #plot the line of the second regression
        

        plt.xlim(start_date_event - pd.Timedelta(end_date_event - start_date_event + '1D'), end_date_event + pd.Timedelta(end_date_event - start_date_event + '1D')) #auto zoom
        plot_height = (regression_interval_end['millimeters'].max() - regression_interval_start['millimeters'].min()) /1.5 #auto zoom
        plt.ylim(regression_interval_start['millimeters'].min()-plot_height, regression_interval_end['millimeters'].max()+plot_height) #auto zoom
        
        
        plt.axvline(x=start_date_event, color='blue', linestyle='--', label='begining of the event', linewidth = 2) #plot a vertical line to represent the begining of the event
        plt.axvline(x=end_date_event, color='green', linestyle='--', label='end of the event', linewidth = 2) #plot a vertical line to represent the end of the event
        
        
        plt.xlabel('Date')
        plt.ylabel('Golojang pressure series water equivalent [mm]')
        plt.title(f"Linear regression on the event {start_date_event}. The precipitation is estimated to {precip_one_event[0]:.2f}")
        plt.legend()
        plt.grid(True)

        plt.show()



#exporting the data to a csv file    
precipitation = pd.DataFrame(total_precip)
precipitation = precipitation.rename(columns = {2 : 'golojang_mm'})
precipitation = precipitation.rename(columns = {1 : 'date_event_end'})
precipitation = precipitation.rename(columns = {0 : 'date_event_start'})
precipitation = precipitation.set_index(precipitation['date_event_start'])
precipitation = precipitation.drop('date_event_start', axis =1)
precipitation.to_csv(r"computed_precipitation_golojang_b.csv")


#computing the precipitation for the period DJFMAM for each year
for year in [2019,2020]:
    print(f'The total precipitation for the period {year}-12-01 to {year+1}-05-31 is {precipitation["golojang_mm"][mask_winter_pre_monsoon(precipitation, year)].values.sum():.0f} mm')
print(f'The total precipitation for the period 2021-12-01 to 2022-04-14 is {precipitation["golojang_mm"][mask_winter_pre_monsoon(precipitation, 2021)].values.sum():.0f} mm')

# print(precipitation)
