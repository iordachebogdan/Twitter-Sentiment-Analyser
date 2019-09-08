from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
import re

tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))

def tweet_cleaner(text: str) -> str:
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.replace(u'ï¿½', '?')
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()

    # remove unnecessary whitespaces
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()
