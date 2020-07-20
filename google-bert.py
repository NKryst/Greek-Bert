import torch
import pandas as pd
from tqdm.notebook import tqdm

df = pd.read_csv(r'C:/Users/nickr/Desktop/Projects/R_Twitter/Twitter/tweets.csv',
names =['id','text','Anger','Disgust','Fear','Happiness','Sadness','Surprise']
)
df.set_index('id', inplace = True)
df.head()
