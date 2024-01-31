import sys
from dataclasses import dataclass
from pathlib import Path
# Adding the project root to the system path for module import
sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np 
import pandas as pd
from src.exception import securelinkException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    # Configuration class for data transformation artifacts
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")


class DataTransformation:
    def __init__(self):
        # Initializing data transformation configuration
        self.data_transformation_config=DataTransformationConfig()

    def initiate_data_transformation(self,train_path,test_path):
        try:            
            # Reading the training and testing datasets
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            # Columns to be excluded from feature set
            input_column_name = ["url","label","result","Unnamed: 0"]

            # Target/label column name
            target_column_name="result"

            input_feature_train_df = train_df.drop(columns=input_column_name, axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=input_column_name, axis=1)
            target_feature_test_df=test_df[target_column_name]  

            logging.info(f"Column dropping perform")   
            # Combining feature and target arrays for train and test datasets         
            train_arr = np.c_[
                input_feature_train_df, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[input_feature_test_df, np.array(target_feature_test_df)]
            # Returning the train and test arrays
            return (                
                train_arr,
                test_arr,
            )
        except Exception as e:
            # Handling exceptions and logging
            raise securelinkException(e,sys)
