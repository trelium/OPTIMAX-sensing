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
    Extracts a valid stream name from the path or filename provided in input. 
    Valid stream names are those specified as keys in the RECORD_MAP dictionary. 
    """
    try:
        streamname = path.split('_')[-2]
    except:
        streamname = '/'
    if streamname in RECORD_MAP.keys():
        return streamname
    else:
        logging.info(f'Filename not recognized at location: {path}')
        return None 

def chunk_file_index(path, streams):
    """navigate all subfolders looking for json files and create list of lists containing locations.
    Only include streams contained in 'streams' variable in the list returned """

    to_analyze = []
    logging.info('Building files index')
    for (root,dirs,files) in tqdm(os.walk(path)):
        for file in files:
            if file.endswith('.json'):
                if sanitize_filename(file) in streams:
                    to_analyze.append(os.path.join(root, file))
    
    return to_analyze


def analyze_files(path,db):
    stream_name = sanitize_filename(path)
    df = sanitize_file(path, stream_name)
    db.insert(data = select_datapoints(df, stream_name), stream = stream_name)


def main():
    logging.basicConfig(filename='preprocessing.log', 
                        level=logging.ERROR,filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        datefmt='%d/%m/%Y %I:%M:%S %p') 
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Directory containing files to be processed. Make sure you have read privileges to it.", type=dir_path)
    parser.add_argument("--streams", help="Specify individual streams for which files should be processed.", nargs="+", default=RECORD_MAP.keys())   
    args = parser.parse_args()

    paths_chunks = chunk_file_index(args.path, args.streams)
    ##get all lists and pass them to pool
    
    print(f'Analyzing {len(paths_chunks)} files')
    db = SensingDB()
    logging.info('Database correctly instantiated')
    for i in tqdm(paths_chunks):
        analyze_files(i, db)
    db.close_connection()
    print('Execution complete.')


if __name__ == '__main__':
    print("""
        .::::     .:::::::  .::: .::::::.::.::       .::      .:       .::      .::
  .::    .::  .::    .::     .::    .::.: .::   .:::     .: ::      .::   .::  
.::        .::.::    .::     .::    .::.:: .:: . .::    .:  .::      .:: .::   
.::        .::.:::::::       .::    .::.::  .::  .::   .::   .::       .::     
.::        .::.::            .::    .::.::   .:  .::  .:::::: .::    .:: .::   
  .::     .:: .::            .::    .::.::       .:: .::       .::  .::   .::  
    .::::     .::            .::    .::.::       .::.::         .::.::      .::
                                                                               
                                                                               """)
    main()

