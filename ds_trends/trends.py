from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from typing import List
from ds_trends.frequency_calculator import FrequencyCalculator

class TopTrends:

    def __init__(self, stop_words=None, bigrams: bool=True, min_freq: float=0.01):

        self.ngram_range = (1,2) if bigrams else (1,1)
        self.min_df = min_freq
        self.stop_words = stop_words
        self.vectorizer = CountVectorizer(ngram_range=self.ngram_range, 
                                              max_df=0.7,
                                              min_df=self.min_df,
                                              stop_words= self.stop_words)
        
        self.freq_calculator = FrequencyCalculator(self.vectorizer)

        
    def is_trend(self, freq_present: float, freq_past: float, rate: float):
        freq_rate = (freq_present-freq_past)/freq_past
        return freq_rate > rate
    
    def get_trends(self, texts_present: List[str], texts_past: List[str], rate: float):
        trends = {}
        freqs_present = self.freq_calculator.get_freqs(texts_present)
        freqs_past = self.freq_calculator.get_freqs(texts_past)
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
    


