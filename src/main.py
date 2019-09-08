#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
cols = ['sentiment', 'id', 'date', 'query_string', 'user', 'text']
data_frame = pd.read_csv(
    './training data/training.1600000.processed.noemoticon.csv',
    header=None,
    names=cols,
    encoding='latin-1'
)
data_frame.drop(['id', 'date', 'query_string', 'user'], axis=1, inplace=True)

#%%
print(data_frame[data_frame.sentiment == 0].head(10))
print(data_frame[data_frame.sentiment == 4].head(10))

#%%
data_frame['pre_clean_len'] = [len(t) for t in data_frame.text]

#%%
from utils.data_dict import data_dict
from pprint import pprint
pprint(data_dict(data_frame))

#%%
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_facecolor((1, 1, 1))
plt.boxplot(data_frame.pre_clean_len)
plt.show()


#%%
print(data_frame[data_frame.pre_clean_len > 140].head(10))


#%%
from bs4 import BeautifulSoup
example1 = BeautifulSoup(data_frame.text[279], 'lxml')
print(len(example1.get_text()))

#%%
print(data_frame.text[343])
import re
print(re.sub(r'@[A-Za-z0-9]+', '', data_frame.text[343]))

#%%
print(data_frame.text[0])
print(re.sub(r'https?://[A-Za-z0-9./]+', '', data_frame.text[0]))


#%%
import codecs

print(data_frame.text[226])
testing = data_frame.text[226].replace(u'ï¿½', '?')
print(testing)


#%%
print(data_frame.text[175])
print(re.sub('[^a-zA-Z]', ' ', data_frame.text[175]))

#%%
from utils.tweet_cleaner import tweet_cleaner

testing = data_frame.text[:100]
test_results = []
for t in testing:
    test_results.append(tweet_cleaner(t))
pprint(test_results)


#%%
print("Cleaning and parsing the tweets...")
clean_tweet_texts = []
num_of_tweets = data_frame.shape[0]
for i in range(0, num_of_tweets):
    if (i+1) % 50000 == 0:
        print("Tweets %d of %d processed" % (i+1, num_of_tweets))
    clean_tweet_texts.append(tweet_cleaner(data_frame.text[i]))

#%%
