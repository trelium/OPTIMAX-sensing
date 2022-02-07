"""
Database population routine 
Optimax project, Jan 2022   
"""

import logging
import sys
import os 
from database import SensingDB
from mappings import RECORD_MAP
import argparse
from multiprocessing import Pool, cpu_count
from scrape_stream_log import *
from tqdm import tqdm 

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

def sanitize_filename(path):
    """
    Extracts a valid stream name from the path provided in input. 
    Valid stream names are those specified as keys in the RECORD_MAP dictionary. 
    """
    streamname = path.split('_')[-2]
    if streamname in RECORD_MAP.keys():
        return streamname
    else:
        logging.error('Filename not recognized at location: {path}')
        return None 

def chunk_file_index(path):
    #navigate all subfolders looking for json files and create list of lists containing locations
    to_analyze = []
    logging.info('Building files index')
    for (root,dirs,files) in tqdm(os.walk(path)):
        for file in files:
            if file.endswith('.json'):
                to_analyze.append(os.path.join(root, file))
    n = cpu_count()
    k, m = divmod(len(to_analyze), n)
    return (to_analyze[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def analyze_files(paths:list, db:SensingDB):
    for file in paths:
        stream_name = sanitize_filename(file)
        try:
            df = sanitize_file(file, stream_name)
            if not db.insert(data = select_datapoints(df, stream_name), stream = stream_name):
                logging.error(f'Data upload failure for file {file}')
        except:
            continue

                    

def main():
    logging.basicConfig(filename='preprocessing.log', 
                        level=logging.DEBUG,filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        datefmt='%d/%m/%Y %I:%M:%S %p') 
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    db = SensingDB()
    logging.info('Database correctly instantiated')

    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Directory containing files to be processed. Make sure you have read privileges to it.", type='dir_path')
    parser.add_argument("--stream", help="Specify individual strams for which files should be processed.", nargs="+", default=RECORD_MAP.keys())   
    args = parser.parse_args()

    ##get all lists and pass them to pool
    with Pool(cpu_count()) as p:
        ...
        #call analyze_files on multiple workers and passing db instance 



    db.close_connection()

if __name__ == 'main':
    main()

