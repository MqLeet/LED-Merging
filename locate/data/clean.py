import json
import pdb
import pandas as pd
from pandas import read_parquet
pd.set_option('display.max_columns', None)

def clean_code(n=500):
    clean_d = []
    with open("143k-Tested-Python-Alpaca-Vezora.json","r") as fr:
        data = json.load(fr)
    for sample in data[:n]:
        # pdb.set_trace()
        if len(sample["input"]) != 0:
            continue
        clean_d.append(sample)
    with open("143k-Tested-Python-Alpaca-Vezora-clean.json","w") as fw:
        json.dump(clean_d, fw, indent=4)
# clean_code()
def read_parquet_gsm8k(file):
    df = read_parquet(file)
    print(df.head(5))
    
read_parquet_gsm8k("./humaneval.parquet")