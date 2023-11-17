from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from typing import List


class TopTrends:
    
    def __init__(self, stop_words=None, bigrams: bool=True, min_freq: float=0.01):

        self.ngram_range = (1,2) if bigrams else (1,1)
        self.min_df = min_freq
        self.stop_words = stop_words
        self.vectorizer = CountVectorizer(ngram_range=self.ngram_range, 
                                              max_df=0.7,
                                              min_df=self.min_df,
                                              stop_words= self.stop_words)
        
    
    def get_freqs(self, texts: List[str]):
        """Get a dictionnary with all the words in a text and their norm freq"""
        X = self.vectorizer.fit_transform(texts)

        bigrams = [word for word in self.vectorizer.get_feature_names_out() if ' ' in word]
        bigram_counts = np.array([self.vectorizer.vocabulary_[bigram] for bigram in bigrams])
        bigram_freqs_ = dict(zip(bigrams, bigram_counts/len(texts)))
        most_freq_bigrams = sorted(bigram_freqs_, reverse=True)[:10]

        word_counts_X = np.array(np.sum(X, axis=0))[0]

        for bigram in most_freq_bigrams:
            tokenized_bigram = bigram.split()
            for token in tokenized_bigram:
                if token in self.vectorizer.get_feature_names_out():
                    word_counts_X[self.vectorizer.vocabulary_[token]] -= self.vectorizer.vocabulary_[bigram]
    
        total_freqs = word_counts_X/len(texts)
        words_freqs_ = dict(zip(self.vectorizer.get_feature_names_out(), total_freqs))
        return words_freqs_
        
    def is_trend(self, freq_present: float, freq_past: float, rate: float):
        freq_rate = (freq_present-freq_past)/freq_past
        return freq_rate > rate
    
    def get_trends(self, texts_present: List[str], texts_past: List[str], rate: float):
        trends = {}
        freqs_present = self.get_freqs(texts_present)
        freqs_past = self.get_freqs(texts_past)
        for word in freqs_present.keys():
            if freqs_present[word] < 0.01:
                continue
            if word in freqs_past.keys():
                if self.is_trend(freqs_present[word], freqs_past[word], rate):
                    trends[word] = (freqs_present[word] - freqs_past[word])/freqs_past[word]
            else:
                if freqs_present[word] > 0.05:
                    trends[word] = 9999
        return trends
    


