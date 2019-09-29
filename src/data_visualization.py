#%%
import pandas as pd
import numpy as np

#%%
clean_csv = '../training data/clean_tweets.csv'
df = pd.read_csv(clean_csv,index_col=0)

#%%
from utils.wordcloud_visualizer import wc_visualizer

neg_tweets = df[df.target == 0]
wc_visualizer(neg_tweets)

pos_tweets = df[df.target == 1]
wc_visualizer(pos_tweets)

#%%
### Count word frequency
from sklearn.feature_extraction.text import CountVectorizer
cvec = CountVectorizer()
cvec.fit(df.text)

#%%
len(cvec.get_feature_names())

#%%
neg_doc_matrix = cvec.transform(df[df.target == 0].text)
pos_doc_matrix = cvec.transform(df[df.target == 1].text)

#%%
neg_tf = np.squeeze(np.asarray(np.sum(neg_doc_matrix, axis=0)))
pos_tf = np.squeeze(np.asarray(np.sum(pos_doc_matrix, axis=0)))

#%%
term_freq_df = \
    pd.DataFrame([neg_tf, pos_tf], columns=cvec.get_feature_names()) \
      .transpose()

#%%
term_freq_df.columns = ['negative', 'positive']
term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive']
# see if everything is alright
term_freq_df.sort_values(by='total', ascending=False).iloc[:10]

#%%
# save the freq for later use
freq_csv = '../training data/word_freq.csv'
term_freq_df.to_csv(freq_csv, encoding='utf-8')
