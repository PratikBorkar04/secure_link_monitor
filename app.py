import pandas as pd
from src.exception import securelinkException
import sys
df = pd.DataFrame([1,2,3,4])
a = 10
try:
    a/0
except Exception as ex:
    raise securelinkException(ex,sys) from ex
