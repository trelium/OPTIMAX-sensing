"""
Database Management Utilities 
Optimax project, Jan 2022   
"""
import os 
import mysql.connector as db
from mysql.connector import errorcode, Error
import logging
from dotenv import load_dotenv
from queries import TABLES

load_dotenv()

logging.basicConfig(filename='database.log') # , encoding='utf-8', level=logging.INFO)
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
