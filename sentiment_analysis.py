# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 21:43:17 2020

@author: tomol
"""
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from nltk import TweetTokenizer
from html import unescape
import spacy
from spacy.lang.en import STOP_WORDS
import gzip
import dill

#read data
df = pd.read_csv("Sentiment Analysis Dataset.csv", error_bad_lines=False)

#tokenizer is a special module to analyse tweets. Transforms tweets to make it more consistent. It removes all tags (@Tomo) and other stuff.
tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

#this converts some html string into readable characters
def preprocessor(doc):
    return unescape(doc.lower())

#spacy allows you to find the root form of every word. Ex: eating --> eat; ate --> eat. This is to identify words more accurately.
nlp = spacy.load("en_core_web_sm", disable=["ner", "par", "tagger"])

def lemmatizer(doc):
    return [word.lemma_ for word in nlp(doc)]

#stop words are characters we want to remove from our dataset when training our model: Ex: it, I, a. Generic words that bring no value to the analysis.
STOP_WORDS_lemma = [word.lemma_ for word in nlp(" ".join(list(STOP_WORDS)))]
STOP_WORDS_lemma = set(STOP_WORDS_lemma).union({",", ".", ";"}) #add few characters to stop words list


#custom step to include the tokenizer and lemmatizer in the data pipeline
class SelectText_Lemma(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, list):
            docs = X
        else:
            docs = X.to_list()
        
        for i, doc in enumerate(docs):
            tweet = tweet_tokenizer.tokenize(doc)
            y = " ".join(tweet)
            docs[i] = y

        for i, doc in enumerate(nlp.pipe(docs, batch_size=20, n_threads=3, disable=["ner", "par", "tagger"])):
            lemma_doc = [tok.lemma_ for tok in doc]
            y = " ".join(lemma_doc)
            docs[i] = y

        return docs


#hashing vectorizer, faster to compute but provides less insight
vectorizer_hash = HashingVectorizer(preprocessor=preprocessor,
                             # tokenizer=lemmatizer,
                             # ngram_range=(1, 2),
                             alternate_sign=False,
                             stop_words=STOP_WORDS_lemma)

#tfidg vectorizer, slow but good at extracting key words
vectorizer_tfidf = TfidfVectorizer(preprocessor=preprocessor, 
                              stop_words=STOP_WORDS_lemma,
                              ngram_range=(1, 2))

#Naive Baye classifier
clf = MultinomialNB()
preprocess = SelectText_Lemma()

#data pipeline
pipe = Pipeline([('preprocess', preprocess), ("vectorizer", vectorizer_tfidf), ("classifier", clf)])

#training dataset
X = df['SentimentText']
y = df['Sentiment']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pipe.fit(X, y)


#save model as a dill to avoid having to train the model every time we want to predict.
with gzip.open("sentiment_model_2.dill.gz", "wb") as f:
    dill.dump(pipe, f, recurse=True)
