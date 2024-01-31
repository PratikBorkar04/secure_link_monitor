import logging
from datetime import datetime
import os

# Directory where log files will be stored
LOG_DIR = "secure_link_monitor_logs"

# Current timestamp for naming the log file
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# Name of the log file including the timestamp
LOG_FILE_NAME = f"log_{CURRENT_TIME_STAMP}.log"

# Create the log directory if it doesn't already exist
os.makedirs(LOG_DIR,exist_ok=True)

# Full path for the log file
LOG_FILE_PATH = os.path.join(LOG_DIR,LOG_FILE_NAME)

logging.basicConfig(filename=LOG_FILE_PATH,
                    filemode='w',
                    format='[%(asctime)s]%(name)s -%(levelname)s - %(message)s',
                    level=logging.INFO
                    )