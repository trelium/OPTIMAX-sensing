"""
Raw Data Converter and json Scraper 
Optimax project, Jan 2022   
"""
import json
import logging 
import pandas as pd
from mappings import RECORD_MAP

#currently ony works for battery stream

def sanitize_file(path, stream_name):
    """
    Takes corrupt file and converts it to pandas dataframe according to the 
    specified mappings  
    """
    try:
        with open(path) as infile:
            content = "[" + infile.read().replace("}\n{", "},\n{") + "]"
            content = content.replace("}{", "},{")
            data = json.loads(content) #list of dictionaries
            out_df = pd.json_normalize(data, RECORD_MAP[stream_name]['record_path'], RECORD_MAP[stream_name]['meta'])
            return out_df
    except:
        logging.warning(f"Could not parse file: {path}")
        return None

def select_datapoints(df):
    """
    Takes in input a dataframe representing a portion of 
    sensor stream information and returns only the relevant information for analysis 
    """
    #return columns 
    return

#apps
#df = sanitize_file('/home/jmocel/trelium/OPTIMAX-sensing/data/optimax_ps_data/2021_files/7_2021-02-11/601c471f3800bbe3308580b9-om_10/601c471f3800bbe3308580b9-om_ActiveApps_1613035924518/601c471f3800bbe3308580b9-om_ActiveApps_1613035924518.json', 'Apps')
#wifi
#df = sanitize_file('/home/jmocel/trelium/OPTIMAX-sensing/data/optimax_ps_data/2021_files/-2_2021-03-02/603b6eb07708a774b94fa0d8-om_01/603b6eb07708a774b94fa0d8-om_WiFi_1614644099658/603b6eb07708a774b94fa0d8-om_WiFi_1614644099658.json', 'WiFi')
#df = sanitize_file('/home/jmocel/trelium/OPTIMAX-sensing/data/optimax_ps_data/2021_files/7_2021-02-07/601c471f3800bbe3308580b9-om_14/601c471f3800bbe3308580b9-om_Light_1612703751977/601c471f3800bbe3308580b9-om_Light_1612703751977.json', 'Light')
df = sanitize_file('/home/jmocel/trelium/OPTIMAX-sensing/data/optimax_ps_data/2021_files/7_2021-02-10/601c471f3800bbe3308580b9-om_15/601c471f3800bbe3308580b9-om_PhysicalActivity_1612967373364/601c471f3800bbe3308580b9-om_PhysicalActivity_1612967373364.json', 'PhisicalActivity')
print(df)
print(df.columns)

