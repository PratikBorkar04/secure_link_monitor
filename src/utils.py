import numpy as np
import pandas as pd
import os
from src.exception import securelinkException
import pickle
import sys

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as ex:
        raise securelinkException(ex, sys)