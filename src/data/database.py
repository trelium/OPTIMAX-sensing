"""
Database Management Utilities 
Optimax project, Jan 2022   
"""
from email import charset
import os
import string 
import mysql.connector as db
from mysql.connector import errorcode, Error
import logging
from dotenv import load_dotenv
from queries import TABLES
from mappings import RECORD_MAP
import re

load_dotenv()

def sanitize_obs(obs):
    obs = str(obs)[1:-1]
    obs = obs.replace('list(','').replace('])',']').replace('None','NULL')
    return obs
    

class SensingDB:
    def __init__(self) -> None:
        try: 
            self.cnx = db.connect(user=os.environ.get('DB_USER'), 
                                password=os.environ.get('DB_PW'),
                                host=os.environ.get('DB_HOST'),
                                port=os.environ.get('DB_PORT'))
            self.cursor = self.cnx.cursor()
            #self.cnx.setencoding('utf-8')
            logging.info(f"Connection to DB established")

        except db.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                logging.error(f"Something is wrong with your user name or password. Connection to DB failed")
    
        #create database in case it does not exist 
        try:
            self.cursor.execute(f"USE {os.environ.get('DB_NAME')}")
        except Error as err_in:
            if err_in.errno == errorcode.ER_BAD_DB_ERROR:   #DB does not exist 
                try:
                    self.cursor.execute(
                        f"CREATE DATABASE {os.environ.get('DB_NAME')} DEFAULT CHARACTER SET 'utf8mb4'")
                    self.cursor.execute(f"""ALTER DATABASE {os.environ.get('DB_NAME')} 
                                            CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci;""")
                except Error as err:
                    print(f"Failed creating database: {err}")
                    logging.warning(f"Creation of DB failed")
                    exit(1)
            else: 
                logging.warning(f'MySQL error:\n{err_in}')

        #check if table is already present, otherwise create table 
        for streamname in TABLES:
            try:
                self.cursor.execute(TABLES[streamname])
                logging.info(f"Created table for: {streamname}")
            except Error as err:
                if err.errno != errorcode.ER_TABLE_EXISTS_ERROR:
                    logging.error(err.msg)
        logging.info('Table existence checks completed')

    def close_connection(self):
        try:
            self.cnx.commit()
            self.cursor.close()
            self.cnx.close()
        except Error as err:
            logging.error(f'Error while finalizing SQL upoad: {err.msg}')

    def insert(self, data:list, stream:string, columns:tuple = None):
        """
        Inserts the observations passed as a list in input to the correct table 
        corresponding to the 'stream' argument. 
        Arguments:
        - stream: name of sensor stream as extracted from filename 
        - data: list of tuples, each of the correct dimension  
        - columns: optional. Specify in chich columns the data should fall into 

        """
        if stream in RECORD_MAP.keys():  #check if there is a table for the stream 
            obs = sanitize_obs(data)
            if columns: 
                qry = f"""INSERT INTO {RECORD_MAP[stream]['target_table']} {columns} VALUES {obs}"""
            else:
                qry = f"""INSERT INTO {RECORD_MAP[stream]['target_table']} VALUES {obs}"""
            
            try:    
                self.cursor.execute(qry)
                return True
            except Error as err:
                logging.error(f"INSERT FAILED for observation: {obs} \nColumns: {columns}\nErrorMsg: {err.msg}")
                return False 
        else:
            logging.error(f"INSERT FAILED: No table created for the given stream: {stream}")
            return False