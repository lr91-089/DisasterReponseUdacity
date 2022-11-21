"""This module is an util script for train_classifier.py."""
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin

nltk.download(['punkt', 'wordnet','stopwords','omw-1.4','averaged_perceptron_tagger'])

def tokenize(text):
    """Tokenizes the messages text."""
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    lower_text = text.lower().strip()
    no_stop_words = re.sub(r"[^a-zA-Z0-9]", " ", lower_text)
    words = word_tokenize(no_stop_words)
    words = [w for w in words if w not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]
    return lemmed

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """Class of the custom transformer."""
    def starting_verb(self, text):
        """Checks if starting verb.

        Keyword arguments:
        text -- text to be checked
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags)>0:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
        return False

    def fit(self, X, y=None):
        """Fits the training data to the model.

        Keyword arguments:
        X -- input values
        y -- target values, default: None
        """
        return self

    def transform(self, X):
        """Transforms the input data with the startingverb transformer.

        Keyword arguments:
        X -- input values
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
