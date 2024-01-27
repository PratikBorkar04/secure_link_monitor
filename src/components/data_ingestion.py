import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs

# Your existing imports for logging and exceptions
from src.logger import logging
from src.exception import securelinkException

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


# DataIngestionConfig remains unchanged
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def extract_url_features(self, url):
        parsed_url = urlparse(url)

        url_length = len(url)
        domain = parsed_url.netloc
        hostname_length = len(domain)

        path = parsed_url.path
        path_length = len(path)

        fd_length = len(url.split('/')[-1])
        count_of_dash = url.count('-')
        count_of_at = url.count('@')
        count_of_question = url.count('?')
        count_of_percent = url.count('%')
        count_of_dot = url.count('.')
        count_of_equal = url.count('=')
        count_of_http = url.count('http')
        count_of_https = url.count('https')
        count_of_www = url.count('www')
        count_of_digits = sum(c.isdigit() for c in url)
        count_of_letters = sum(c.isalpha() for c in url)
        count_of_dir = url.count('/')

        use_of_ip = 1 if domain.replace('.', '').isdigit() else 0

        # Additional columns
        qty_hyphen_url = url.count('-')
        length_url = len(url)
        qty_tilde_url = url.count('~')
        qty_dot_url = url.count('.')
        qty_percent_url = url.count('%')
        length_domain = len(domain)
        params_length = len(parse_qs(parsed_url.query))
        qty_and_params = url.count('&')
        qty_hyphens_params = url.count('-')
        directory_length = len(parsed_url.path.split('/'))
        qty_equal_params = url.count('=')
        qty_equal_url = url.count('=')
        qty_slash_url = url.count('/')
        qty_slash_directory = url.count('/') - 1
        file_length = len(parsed_url.path.split('/')[-1])
        qty_and_url = url.count('&')
        qty_dot_params = url.count('.')

        return [hostname_length, path_length, fd_length, count_of_dash,
                count_of_at, count_of_question, count_of_percent, count_of_dot,
                count_of_equal, count_of_http, count_of_https, count_of_www,
                count_of_digits, count_of_letters, count_of_dir, use_of_ip,
                qty_hyphen_url, length_url, qty_tilde_url, qty_dot_url,
                qty_percent_url, length_domain, params_length, qty_and_params,
                qty_hyphens_params, directory_length, qty_equal_params,
                qty_equal_url, qty_slash_url, qty_slash_directory, file_length,
                qty_and_url, qty_dot_params]

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method")
        try:
            df = pd.read_csv("https://raw.githubusercontent.com/PratikBorkar04/linksafetyscanner/main/linksafety/datasets/urldata.csv")
            logging.info("Read the data")

            # Define new columns
            new_columns = ['hostname_length', 'path_length', 'fd_length', 'count_of_dash',
                           'count_of_at', 'count_of_question', 'count_of_percent', 'count_of_dot',
                           'count_of_equal', 'count_of_http', 'count_of_https', 'count_of_www',
                           'count_of_digits', 'count_of_letters', 'count_of_dir', 'use_of_ip',
                           'qty_hyphen_url', 'length_url', 'qty_tilde_url',
                           'qty_dot_url', 'qty_percent_url', 'length_domain', 'params_length',
                           'qty_and_params', 'qty_hyphens_params', 'directory_length',
                           'qty_equal_params', 'qty_equal_url', 'qty_slash_url',
                           'qty_slash_directory', 'file_length', 'qty_and_url', 'qty_dot_params']

            # Apply URL feature extraction
            df[new_columns] = df['url'].apply(self.extract_url_features).apply(pd.Series)
            logging.info("New Columns Created")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train Test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion Completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path 
            )
        except Exception as ex:
            raise securelinkException(ex, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)