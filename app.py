import pandas as pd
from src.exception import securelinkException
import sys
df =pd.read_csv("src/datasets/urldata.csv")
print(df.head())
