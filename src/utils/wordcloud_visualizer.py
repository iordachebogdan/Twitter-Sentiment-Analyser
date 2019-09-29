from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

def wc_visualizer(tweets: pd.DataFrame) -> None:
    string = pd.Series([t for t in tweets.text]).str.cat(sep=' ')
    wordcloud = WordCloud(width=1600, height=800, max_font_size=200) \
                .generate(string)
    plt.figure(figsize=(12,10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
