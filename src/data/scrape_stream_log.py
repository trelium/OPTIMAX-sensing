"""
Raw Data Converter and json Scraper 
Optimax project, Jan 2022   
"""
import json
import logging 
import pandas as pd
from mappings import RECORD_MAP
from pprint import pprint
import sys 


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
        logging.error(f"Could not insert observations present in file (Could not parse file): {path}")
        return None

def select_datapoints(df, stream_name):
    """
    Takes in input a dataframe representing a portion of 
    sensor stream information and returns only the relevant information for analysis 
    """
    #check if in column names there are all of the columns_sel, otherwise do create those that are missing and set null values
    #if set(df.columns) != set(RECORD_MAP[stream_name]['columns_sel']) and stream_name != 'PhysicalActivity':
    if df is not None:
        missing_cols = set(RECORD_MAP[stream_name]['columns_sel']) - set(df.columns)
    else:
        return None
    
    exceptions = set(['PhysicalActivity', 'Accelerometer'])
    if len(missing_cols) != 0  and stream_name not in exceptions:
        try:
            missing_cols = list(missing_cols)
            df[missing_cols] = None
            logging.info(f"The following columns of stream {stream_name} were set to null: {missing_cols}")
        except:
            logging.error(f"Refactoring of {stream_name} failed. \nDataframe:\n{df}")

    if stream_name == 'PhysicalActivity':
        #has messy schema, needs reordering 
        #'senseStartTimeMillis', 'userid', 'activityName', 'activityType', 'confidence' is target schema   
        activity_map = {'in_vehicle':0, 'on_bicycle':1, 'on_foot':2, 'still':3, 'unknown':4, 'tilting':5, 'walking':6, 'running':7}
        try: 
            df.columns = df.columns.str.lower()
            missing_cols = set(activity_map.keys()) - set(df.columns) 
            if len(missing_cols) != 0:
                missing_cols = list(missing_cols)
                df[missing_cols] = None
            df = pd.melt(df, id_vars=['sensestarttimemillis', 'userid'], 
                    value_vars=list(activity_map.keys()), 
                    var_name='activityname', 
                    value_name='confidence')
            df['activitytype'] = [activity_map[key] for key in df['activityname']]
        except:
            logging.error(f"Refactoring of {stream_name} failed. \nDataframe:\n{df}")

    elif stream_name == 'Accelerometer':
        try:
            acc_new = pd.DataFrame({'sensorTimeStamps' : df['sensorTimeStamps'][0],
                        'xAxis' : df['xAxis'][0],
                        'yAxis' : df['yAxis'][0],
                        'zAxis' : df['zAxis'][0]
                        })
            acc_new['userid'] = df['userid'][0]
            df = acc_new
        except: #ValueError was written prior, KeyError is correct for missing key
            logging.error('Inconsistencies in dimension of records for Accelerometer stream')
            df = None
    
    try:
        df = df[RECORD_MAP[stream_name]['columns_sel']]
        records = df.to_records(index=False)  #return rows as list of tuples
        return list(records)
    except:
        logging.error(f"Conversion to records of stream {stream_name} failed. \nDataframe:\n{df}")
        return None

