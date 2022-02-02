"""
Database Management Utilities 
Optimax project, Jan 2022   
"""
import os
import string 
import sys
import mysql.connector as db
from mysql.connector import errorcode, Error
import logging
from dotenv import load_dotenv
from queries import TABLES

load_dotenv()
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.basicConfig(filename='preprocessing.log') # , encoding='utf-8', level=logging.INFO)
                    #format='%(asctime)s - %(levelname)s: %(message)s') #, datefmt='%d/%m/%Y %I:%M:%S %p'

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
                print("Something is wrong with your user name or password")
                logging.warning(f"Connection to DB failed")
    
        #create database in case it does not exist 
        try:
            self.cursor.execute(f"USE {os.environ.get('DB_NAME')}")
        except Error as err_in:
            if err_in.errno == errorcode.ER_BAD_DB_ERROR:   #DB does not exist 
                try:
                    self.cursor.execute(
                        f"CREATE DATABASE {os.environ.get('DB_NAME')} DEFAULT CHARACTER SET 'utf8'")
                except Error as err:
                    print(f"Failed creating database: {err}")
                    logging.warning(f"Creation of DB failed")
                    exit(1)
            else: 
                logging.warning(err_in)

        #check if table is already present, otherwise create table 
        for qry in TABLES:
            try:
                print("Creating table: ", qry)
                self.cursor.execute(TABLES[qry])
            except Error as err:
                if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                    print("already exists.")
                else:
                    print(err.msg)
            else:
                print("OK")

    def close_connection(self):
        self.cursor.close()
        self.cnx.close()

    def insert(self, data:list, stream:string, columns:tuple = None):
        """
        Inserts the observations passed as a list in input to the correct table 
        corresponding to the 'stream' argument. 
        Arguemnts:
        - stream: name of sensor stream as extracted from filename 
        - data: list of tuples, each of the correct dimension  
        - columns: optional. Specify in chich columns the data should fall into 

        """
        if stream in TABLES.keys():  #check stream 
            obs = str(data)[1:-1]
            if columns: 
                qry = f"""INSERT INTO {stream} {columns} VALUES {obs}"""
            else:
                qry = f"""INSERT INTO {stream} VALUES {obs}"""
            
            try:    
                self.cursor.execute(qry)
                return True
            except:
                logging.warning(f"INSERT FAILED for observation: {obs} \nColumns: {columns}")
                return False 
        else:
            logging.warning(f"INSERT FAILED: {stream} is not accepted as column name")
            return False