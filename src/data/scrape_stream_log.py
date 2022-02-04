"""
Raw Data Converter and json Scraper 
Optimax project, Jan 2022   
"""
import json
import logging 
import pandas as pd
from mappings import RECORD_MAP


def sanitize_file(path, stream_name):
    """
    Takes a (potentially) corrupt file and converts it to pandas dataframe according to the 
    specified mappings  
    """
    try:
        with open(path) as infile:
            content = "[" + infile.read().replace("}\n{", "},\n{") + "]"
            content = content.replace("}{", "},{")
            data = json.loads(content) #list of dictionaries
            out_df = pd.json_normalize(data, RECORD_MAP[stream_name]['frame']['record_path'], RECORD_MAP[stream_name]['frame']['meta'])
            return out_df
    except:
        logging.warning(f"Could not parse file: {path}")
        return None

def select_datapoints(df, stream_name):
    """
    Takes in input a dataframe representing a portion of 
    sensor stream information and returns only the relevant information for analysis 
    """
    #check if in column names there are all of the columns_sel, otherwise do create those that are missing and set null values
    if set(df.columns) != set(RECORD_MAP[stream_name]['columns_sel']):
        missing_cols = set(df.columns).difference(RECORD_MAP[stream_name]['columns_sel'])
        missing_cols = list(missing_cols)
        df[missing_cols] = None
        logging.info(f"The following columns of stream {stream_name} were set to null: {missing_cols}")

    if stream_name == 'PhysicalActivity':
        #has messy schema, needs reordering 
        #'senseStartTimeMillis', 'userid', 'activityName', 'activityType', 'confidence' is target schema   


    df = df[RECORD_MAP[stream_name]['columns_sel']]
    #return columns 
    return

#apps
#df = sanitize_file('/home/jmocel/trelium/OPTIMAX-sensing/data/optimax_ps_data/2021_files/7_2021-02-11/601c471f3800bbe3308580b9-om_10/601c471f3800bbe3308580b9-om_ActiveApps_1613035924518/601c471f3800bbe3308580b9-om_ActiveApps_1613035924518.json', 'Apps')
#wifi
#df = sanitize_file('/home/jmocel/trelium/OPTIMAX-sensing/data/optimax_ps_data/2021_files/-2_2021-03-02/603b6eb07708a774b94fa0d8-om_01/603b6eb07708a774b94fa0d8-om_WiFi_1614644099658/603b6eb07708a774b94fa0d8-om_WiFi_1614644099658.json', 'WiFi')
#df = sanitize_file('/home/jmocel/trelium/OPTIMAX-sensing/data/optimax_ps_data/2021_files/7_2021-02-07/601c471f3800bbe3308580b9-om_14/601c471f3800bbe3308580b9-om_Light_1612703751977/601c471f3800bbe3308580b9-om_Light_1612703751977.json', 'Light')
#df = sanitize_file('/home/jmocel/trelium/OPTIMAX-sensing/data/optimax_ps_data/2021_files/7_2021-02-10/601c471f3800bbe3308580b9-om_15/601c471f3800bbe3308580b9-om_PhysicalActivity_1612967373364/601c471f3800bbe3308580b9-om_PhysicalActivity_1612967373364.json', 'PhisicalActivity')
df = sanitize_file('/home/jmocel/trelium/OPTIMAX-sensing/data/optimax_ps_data/2021_files/7_2021-02-10/601c471f3800bbe3308580b9-om_15/601c471f3800bbe3308580b9-om_Accelerometer_1612967938706/601c471f3800bbe3308580b9-om_Accelerometer_1612967938706.json', 'Accelerometer')
print(df)
print(df.columns)

