import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.logger import logging
from src.exception import securelinkException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method")
        try:
            df = pd.read_csv("https://raw.githubusercontent.com/PratikBorkar04/linksafetyscanner/main/linksafety/datasets/urldata.csv")
            logging.info("Read the data")
           
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index = False,header=True)
           
            logging.info("Train Test split initiated")
            train_set,test_set = train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index = False,header=True)
            logging.info("Ingestion Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path 
            )
        except Exception as ex:
            raise securelinkException(ex,sys)

if __name__=='__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()
     