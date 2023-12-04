import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer


class FrequencyCalculator():

    def __init__(self, vectorizer):

        self.vectorizer = vectorizer

    def fit_vectorizer(self, texts):
        """
        Fit a Count Vectorizer to a text
        """
        return self.vectorizer.fit_transform(texts)
    
    def get_most_frequent_bigrams(self, texts, first_n_elements=10):
        '''
        get the n most frequent bigrams

        parameters:
        - first_n_elements: number of bigrams to take, default 10

        returns: a list with the n most frequent bigrams ranked from highest to lowest frequency

        '''
        bigrams = [word for word in self.vectorizer.get_feature_names_out() if ' ' in word]
        bigram_counts = np.array([self.vectorizer.vocabulary_[bigram] for bigram in bigrams])
        bigram_freqs_ = dict(zip(bigrams, bigram_counts/len(texts)))

        return sorted(bigram_freqs_, key=bigram_freqs_.get, reverse=True)[:first_n_elements]
    
    def update_unigram_counts(self, texts):
        """
        update unigram counts based on bigram counts.

        parameters:
        - texts: original text data.

        returns: updated word counts array.
        """
        X = self.fit_vectorizer(texts)

        word_counts_X = np.array(np.sum(X, axis=0))[0]

        for bigram in self.get_most_frequent_bigrams(texts=texts):

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
        return self.words_freqs_