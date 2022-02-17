"""
Database population routine 
Optimax project, Jan 2022   
"""

import logging
import sys
import os
from timeit import repeat 
from database import SensingDB
from mappings import RECORD_MAP
import argparse
from multiprocessing import Pool, cpu_count
from scrape_stream_log import *
from tqdm import tqdm 
import pickle
from  itertools import repeat

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
    for (root,dirs,files) in os.walk(path):
        currentdir = root.split(os.path.sep)[-1]
        print(f"Analyzig sub-directory: {currentdir}",  end='\r')
        for file in files:
            if file.endswith('.json'):
                if sanitize_filename(file) in streams:
                    to_analyze.append(os.path.join(root, file))
    print()
    with open('allpaths.pkl', 'wb') as f:
        pickle.dump(to_analyze, f)
    return to_analyze

def get_paths_from_file(path, prefix = None):
    allpaths = []
    with open(path, 'r') as f:
        for line in f:
            if prefix:
                name = prefix + line[1:]
            else: 
                name = line
            allpaths.append(name)
    return allpaths

def analyze_files(path,db):
    stream_name = sanitize_filename(path)
    df = sanitize_file(path, stream_name)
    if df is not None:
        if not db.insert(data = select_datapoints(df, stream_name), stream = stream_name):
            logging.error(f'Could not insert observations present in file: {path}')

def main():
    logging.basicConfig(filename='preprocessing.log', 
                        level=logging.DEBUG,filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        datefmt='%d/%m/%Y %I:%M:%S %p') 
    #logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Directory containing files to be processed. Make sure you have read privileges to it.", type=dir_path)
    parser.add_argument("--streams", help="Specify individual streams for which files should be processed.", nargs="+", default=RECORD_MAP.keys()) 
    parser.add_argument("--pickle", help="Path to binarized list containing paths to files to be analyzed")
    parser.add_argument("--paths_file", help="Path to file containing paths to json files to be analyzed, one per line.")
  
    args = parser.parse_args()

    if args.pickle is not None:
        paths_chunks = pickle.load(open(args.pickle,'rb'))
    elif args.paths_file is not None:
        paths_chunks = get_paths_from_file(path = args.paths_file, prefix='/home/jmocel/trelium/OPTIMAX-sensing/data/optimax_ps_data')
    else:
        paths_chunks = chunk_file_index(args.path, args.streams)

    logging.info(f'Found {len(paths_chunks)} files for the selected streams')
    print(f'Analyzing {len(paths_chunks)} files')
    db = SensingDB()
    logging.info('Database correctly instantiated')
    for i in tqdm(paths_chunks):
        analyze_files(i, db)   
    db.close_connection()   
    logging.info(f'Execution complete for path: {args.path}')


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

