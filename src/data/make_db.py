"""
Database population routine 
Optimax project, Jan 2022   
"""

from lib2to3.pgen2.token import NEWLINE
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
    for (root,dirs,files) in os.walk(path):
        currentdir = root.split(os.path.sep)[-1]
        print(f"Analyzig sub-directory: {currentdir}",  end='\x1b[1K\r')
        for file in files:
            if file.endswith('.json'):
                if sanitize_filename(file) in streams:
                    to_analyze.append(os.path.join(root, file))
    print()
    #return to_analyze
    n = cpu_count()
    k, m = divmod(len(to_analyze), n)
    return len(to_analyze), (to_analyze[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    
    

def analyze_files(paths:list):
    db = SensingDB()
    logging.info('Database correctly instantiated')
    for file in paths: #paths is in fact string type, file is in fact a character
        stream_name = sanitize_filename(file)
        try:
            df = sanitize_file(file, stream_name)
            if not db.insert(data = select_datapoints(df, stream_name), stream = stream_name):
                logging.error(f'Data upload failure for file {file}')
            else:
                return 1
        except:
            continue
    db.close_connection()

                    

def main():
    logging.basicConfig(filename='preprocessing.log', 
                        level=logging.DEBUG,filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        datefmt='%d/%m/%Y %I:%M:%S %p') 
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Directory containing files to be processed. Make sure you have read privileges to it.", type=dir_path)
    parser.add_argument("--streams", help="Specify individual streams for which files should be processed.", nargs="+", default=RECORD_MAP.keys())   
    args = parser.parse_args()

    total, paths_chunks = chunk_file_index(args.path, args.streams)
    #print(type(paths_chunks))
    #print(type(paths_chunks[3]))
    #print(paths_chunks[3])
    ##get all lists and pass them to pool
    with Pool(cpu_count()) as p:
        print(f'Analyzing {total} files:')
        #for _ in tqdm(p.imap(analyze_files,paths_chunks), total = len(paths_chunks)):
        #    pass
        r = list(tqdm(p.map(analyze_files,paths_chunks), total = total))
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

