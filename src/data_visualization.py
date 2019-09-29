#%%
import pandas as pd

#%%
clean_csv = '../training data/clean_tweets.csv'
df = pd.read_csv(clean_csv,index_col=0)

#%%
from utils.wordcloud_visualizer import wc_visualizer

neg_tweets = df[df.target == 0]
wc_visualizer(neg_tweets)

pos_tweets = df[df.target == 1]
wc_visualizer(pos_tweets)
