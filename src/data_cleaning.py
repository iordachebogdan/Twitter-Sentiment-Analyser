#%%
import pandas as pd

#%%
cols = ['sentiment', 'id', 'date', 'query_string', 'user', 'text']
data_frame = pd.read_csv(
    '../training data/training.1600000.processed.noemoticon.csv',
    header=None,
    names=cols,
    encoding='latin-1'
)
data_frame.drop(['id', 'date', 'query_string', 'user'], axis=1, inplace=True)

#%%
data_frame['pre_clean_len'] = [len(t) for t in data_frame.text]

#%%
from utils.data_dict import data_dict
from pprint import pprint
pprint(data_dict(data_frame))

#%%
from utils.tweet_cleaner import tweet_cleaner

print("Cleaning and parsing the tweets...")
clean_tweet_texts = []
num_of_tweets = data_frame.shape[0]
for i in range(0, num_of_tweets):
    if (i+1) % 50000 == 0:
        print("Tweets %d of %d processed" % (i+1, num_of_tweets))
    clean_tweet_texts.append(tweet_cleaner(data_frame.text[i]))


#%%
clean_df = pd.DataFrame(clean_tweet_texts, columns=['text'])
clean_df['target'] = data_frame.sentiment
print(clean_df.head())

#%%
clean_csv = '../training data/clean_tweets.csv'
clean_df.to_csv(clean_csv, encoding='utf-8')

#%%
my_df = pd.read_csv(clean_csv,index_col=0)
my_df.head()

#%%
my_df.info()

#%%
df = pd.read_csv(
    '../training data/training.1600000.processed.noemoticon.csv',
    header=None,
    names=cols,
    encoding='latin-1'
)

#%%
import numpy as np
np.sum(my_df.isnull().any(axis=1))

#%%
df.iloc[my_df[my_df.isnull().any(axis=1)].index, :].head().text

#%%
# Drop null columns
my_df.dropna(inplace=True)
my_df.reset_index(inplace=True, drop=True)
my_df.info()

#%%
my_df.to_csv(clean_csv, encoding='utf-8')