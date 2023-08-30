from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class TopTrends:
    
    def __init__(self, stop_words=None, bigrams=True, min_freq=0.01):

        self.ngram_range = (1,2) if bigrams else (1,1)
        self.min_df = min_freq
        self.stop_words = stop_words
        self.vectorizer = CountVectorizer(ngram_range=self.ngram_range, 
                                              max_df=0.7,
                                              min_df=self.min_df,
                                              stop_words= self.stop_words)
        
    def get_freqs(self, texts):
        """Get a dictionnary with all the words in a text and their norm freq"""
        X = self.vectorizer.fit_transform(texts)
        total_freqs = np.array(np.sum(X, axis=0)/len(texts))[0]
        words_freqs_ = dict(zip(self.vectorizer.get_feature_names_out(), total_freqs))
        return words_freqs_
        
    def is_trend(self, freq_present, freq_past, rate):
        freq_rate = (freq_present-freq_past)/freq_past
        return freq_rate > rate
    
    def get_trends(self, texts_present, texts_past, rate):
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
    


