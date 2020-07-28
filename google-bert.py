import torch
import pandas as pd
from tqdm.notebook import tqdm

data = open('C:\Users\nickr\Desktop\Projects\R_Twitter\Twitter\tweets.csv')
data.encode('utf-8').strip()
df = pd.read_csv(r'C:\Users\nickr\Desktop\Projects\R_Twitter\Twitter\tweets.csv',
names =['id','text','Anger','Disgust','Fear','Happiness','Sadness','Surprise']
)
df = pd.read_csv(r'C:\Users\nickr\Desktop\Projects\R_Twitter\Twitter\tweets.csv',
names =['id','text','Anger','Disgust','Fear','Happiness','Sadness','Surprise']
)
df.set_index('id', inplace = True)
df.head()
