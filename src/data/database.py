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
import pandas as pd

load_dotenv()

def sanitize_obs(obs):
    obs = str(obs)[1:-1]
    obs = obs.replace('list(','').replace('])',']').replace('None','NULL')
    obs = obs.replace('[]', 'JSON_ARRAY()')
    obs = obs.replace('nan','NULL')
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

    def query(self, qry):
        self.cursor.execute(qry)

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
    
    def get_all_timestamps(self, table:string):
        """
        Warning:inefficient on large tables 
        Returns the device_id and timestamp columns from the selected table in the form of a pandas dataframe 
        Arguments:
        - table: string. Name of the target table from which to extract the requested information 
        """
        qry = f"""SELECT TIMESTAMP, DEVICE_ID FROM {table}"""
        try:    
            self.cursor.execute(qry)
            ret = self.cursor.fetchall()
            ret = pd.DataFrame(ret, columns=['TIMESTAMP', 'DEVICE_ID'])
            return ret
        except Error as err:
            logging.error(f"select failed for table {table}\nErrorMsg: {err.msg}")
            return False 

    def get_all_device_ids(self):
        """Returns a list of all device ids present in the database (across all tables)"""
        #all_tables = str([RECORD_MAP[i]['target_table'] for i in RECORD_MAP]).replace("'", "").replace("[","").replace("]","")
        #qry = f"""SELECT DISTINCT DEVICE_ID FROM {all_tables};""" #SELECT city FROM tableA UNION SELECT city FROM tableB UNION SELECT city FROM tableC
        
        all_tables = [RECORD_MAP[i]['target_table'] for i in RECORD_MAP]
        qry = str()
        for i in all_tables:
            qry += f"SELECT DEVICE_ID FROM {i} UNION "
        qry = qry[:-6]
        self.cursor.execute(qry)
        ret = self.cursor.fetchall()
        return ret

    def get_sensing_timespan(self, table = None, exact = True):
        """Returns the start and end date of sensing for each participant in the database, across all tables present 
        If exact is set to true, return all daily time windows for which there is passive sensing data.
        Returns list of tuples 
        """
        all_tables = [RECORD_MAP[i]['target_table'] for i in RECORD_MAP]
        qry = str()
        if table: 
            if exact == False:
                qry = f"""SELECT DEVICE_ID, FROM_UNIXTIME(min(TIMESTAMP/1000)) min, FROM_UNIXTIME(max(TIMESTAMP/1000)) max
                        FROM `{table}` GROUP BY DEVICE_ID """
            else:
                qry = f"""WITH t AS (
                            with tt as (
                            select device_id,substr(from_unixtime(TIMESTAMP/1000),1,10) dd
                            FROM {table}
                            GROUP BY device_id,substr(from_unixtime(TIMESTAMP/1000),1,10)
                            order by 1,2
                            )
                            SELECT device_id,dd as d,ROW_NUMBER() OVER(ORDER BY device_id,dd) i,DATE_SUB(dd,INTERVAL ROW_NUMBER() OVER(ORDER BY device_id,dd) DAY ) as ii
                            FROM tt
                            GROUP BY device_id,dd
                            )
                            SELECT device_id,MIN(d),MAX(d)
                            FROM t
                            GROUP BY device_id,ii;"""
        else: #fetch min and max across all different tables     
            for i in all_tables:
                qry += f"""SELECT DEVICE_ID, FROM_UNIXTIME(min(TIMESTAMP/1000)) min , FROM_UNIXTIME(max(TIMESTAMP/1000)) max
                            FROM `{i}` GROUP BY DEVICE_ID UNION """
            qry = qry[:-6]
            qry = """WITH t AS ( """ + qry + """ ) SELECT DEVICE_ID, min, max from t GROUP BY DEVICE_ID"""
        
        self.cursor.execute(qry)
        ret = self.cursor.fetchall()
        return ret

    def get_android_list(self):
        """Returns a list of device_id that can certainly be associated with android devices (not exhaustive)"""
        
        qry = """SELECT DEVICE_ID FROM PHONE_LIGHT UNION SELECT DEVICE_ID FROM PHONE_APPLICATIONS"""
        self.cursor.execute(qry)
        ret = self.cursor.fetchall()
        return [i[0] for i in ret]