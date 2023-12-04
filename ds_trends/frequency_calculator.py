import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class FrequencyCalculator():

    def __init__(self, stop=None, **kwargs):
        self.stop = stop
        self.stop2 = kwargs.get('stop_words', None)
        if self.stop2 is not None:
            self.stop.extend(self.stop2)
        self.vectorizer = CountVectorizer(
            ngram_range=kwargs.get('ngram_range', (1, 2)),
            max_df=kwargs.get('max_df', 0.5),
            min_df=kwargs.get('min_df', 0.001),
            stop_words=self.stop
        )
        self.words_freqs_ = {}
        self.top_terms = {}
        self.top_lemmas = {}
        self.categories = {}

    def fit_vectorizer(self, texts):
        """
        Fit a Count Vectorizer to a text
        """
        X = self.vectorizer.fit_transform(texts)
        return X
    
    def update_unigram_counts(self, texts):
        """
        update unigram counts based on bigram counts.

        parameters:
        - texts: original text data.

        returns: updated word counts array.
        """
        X = self.fit_vectorizer(texts)

        bigrams = [word for word in self.vectorizer.get_feature_names_out() if ' ' in word]
        bigram_counts = np.array([self.vectorizer.vocabulary_[bigram] for bigram in bigrams])
        bigram_freqs_ = dict(zip(bigrams, bigram_counts/len(texts)))
        most_freq_bigrams = sorted(bigram_freqs_, reverse=True)[:10]

        word_counts_X = np.array(np.sum(X, axis=0))[0]

        for bigram in most_freq_bigrams:

            tokenized_bigram = bigram.split()
            bigram_count = self.vectorizer.vocabulary_[bigram]

            for token in tokenized_bigram:
                if token in self.vectorizer.get_feature_names_out():
                    word_counts_X[self.vectorizer.vocabulary_[token]] -= bigram_count

        return word_counts_X
    
    def get_freqs(self, texts):
        """Get a dictionary with all the words in a text and their norm freq"""
        X = self.fit_vectorizer(texts)
        word_counts_X = self.update_unigram_counts(texts)
        total_freqs = word_counts_X / len(texts)
        words_freqs_ = dict(zip(self.vectorizer.get_feature_names_out(), total_freqs))
        self.words_freqs_ = words_freqs_
        return words_freqs_