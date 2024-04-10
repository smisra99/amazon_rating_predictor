import pandas as pd
import gzip
import json
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('dataset/Clothing_Shoes_and_Jewelry.json.gz')
df = df.drop('verified', 1)
df = df.drop('reviewTime', 1)
df = df.drop('reviewerID', 1)
df = df.drop('asin', 1)
df = df.drop('reviewerName', 1)
df = df.drop('unixReviewTime', 1)
df = df.drop('image', 1)
df.to_csv("Clothing_Shoes_and_Jewelry.csv",index=False)
df=df.sample(frac=0.1,replace=False)
df.to_csv("Clothing_Shoes_and_Jewelry_10.csv",index=False)
