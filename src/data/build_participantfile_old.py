import pandas as pd
import datetime


GESAMTFILE = '/opt/data/optimax_sensing/Gesamtfile_EMA_Baseline_bereinigt_29.01.22_MD.xlsx'

df = pd.read_excel(GESAMTFILE, engine='openpyxl')
df = df[['VP','Date_Time_last_action (MAX) / completed (SEMA3)','Teilnehmer Mattermost']]

#Generate placeholder values for missing columns 
df[['fitbit_id','empatica_id']] = None
df['platform'] = 'infer'
df['label'] = df['VP']
df['end_date'] = None #df['Date_Time_last_action (MAX) / completed (SEMA3)']

#rename and order columns
df = df[['Teilnehmer Mattermost','fitbit_id','empatica_id','VP','platform','label','Date_Time_last_action (MAX) / completed (SEMA3)','end_date']]
df.columns = ['device_id','fitbit_id','empatica_id','pid','platform','label','start_date','end_date']

#drop NAs in pid column 
df = df.dropna(subset=['pid'])
df = df.dropna(subset=['device_id'])

#Convert dtypes
df = df.astype({"label": int, "pid": int})
df['start_date'] = pd.to_datetime(df['start_date'], infer_datetime_format = True)
df['start_date'] = df['start_date'].dt.round('D') #round to day #TODO: rounds to next day in some cases



#group by participant and get first and last date, put in last two columns 
end_dates = df.groupby('pid')['start_date'].max().to_frame()
start_dates = df.groupby('pid')['start_date'].min().to_frame()


df = df.drop_duplicates(subset=['pid']) #delete duplicate rows 

df = df.drop(['start_date','end_date'], axis = 1)

df = df.merge(start_dates, how = 'inner', on = ['pid'])
df = df.merge(end_dates, how = 'inner', on = ['pid'])

#rename columns
df.columns = ['device_id','fitbit_id','empatica_id','pid','platform','label','start_date','end_date']

#save

df.to_csv('/opt/data/optimax_sensing/BaselineParticipantsOptimax.csv')