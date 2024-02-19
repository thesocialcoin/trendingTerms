import numpy as np
from collections import Counter
import pandas as pd
import operator

from sklearn.feature_extraction.text import CountVectorizer

class top_terms_extractor:
        """A class to detect top terms and other features from text

        Attributes:
        stop_words: stop words that can be excluded depending on the project
        ngram_range, max_df, min_df: arguments for CountVectorizer
        vectorizer: An instance from CountVectorizer to count the freqs of words

        words_freqs_: A dictionnary with words and norm freqs (use get_words_freqs to get it)
        top_terms: A dictionnary with top terms and norm freqs (use compute_top_terms to get it)
        top_lemmas: A dictionnary with top lemmas and norm freqs (use compute_top_terms with spacy lemmatizer)
        categories: A dictionnary with categories and the top words representing the categories (get_words_freqs_by_category)
        """
        def __init__(self, **kwargs):
            self.vectorizer = CountVectorizer(
                ngram_range=kwargs.get('ngram_range',(1,2)), 
                max_df=kwargs.get('max_df', 0.5),
                min_df=kwargs.get('min_df', 0.001),
                stop_words=kwargs.get('stop_words', [])
            )
            
            self.words_freqs_ = {}
            self.top_terms={}
            self.top_lemmas={}
            self.categories = {}


        def fit_vectorizer(self, texts):
            """Fit a Count Vectorizer to a text"""
            X = self.vectorizer.fit_transform(texts)
            return X
            
        def get_words_freqs(self, texts):
            """Get a dictionnary with all the words in a text and their norm freq"""
            X = self.fit_vectorizer(texts)

            total_counts = np.array(np.sum(X, axis=0))[0]
            total_freqs = total_counts/len(texts)
            words_freqs_ = [{
                "term": term,
                "count": count,
                "freq": freq
            } for term, count, freq in zip(self.vectorizer.get_feature_names_out(), total_counts, total_freqs)]
            self.words_freqs_ = words_freqs_
            return words_freqs_

        def compute_top_terms(self, n=100, texts=None, nlp=None):
            """
            Computes top terms from a text
            Input:
                - n: number of top terms to output
                - texts: List of texts to extract the topterms 
                - nlp, optional: Spacy model to lemmatize in case we want the top lemmas
            
            Output:
                - top terms
                - top lemmas if nlp!= None  
            """
            if texts!=None:
                self.get_words_freqs(texts)
            if texts==None and self.words_freqs_=={}:
                raise Exception("No dict of words and freqs, you need to pass list of texts.")
            top_terms = sorted(self.words_freqs_, key=lambda item: item["freq"], reverse=True)[:10*n]

            for term in top_terms:
                for term2 in top_terms:
                    if (term["term"] + ' ' in term2["term"]) or (' ' + term["term"] in term2["term"]):
                        term["freq"] = term["freq"] - term2["freq"]
            self.top_terms = sorted(top_terms, key=lambda item: item["freq"], reverse=True)[:n]
            # TODO: Some changes in the top_terms format may break the following code. But it is not used anywhere.
            # should be reviewed if lemmas ara calculated.
            # if nlp!=None:
            #     top_terms = lemma_dict(top_terms, nlp)
            #     self.top_lemmas = dict(sorted(top_terms.items(), key=lambda item: item[1], reverse=True)[:n])
            return sorted(top_terms, key=lambda item: item["freq"], reverse=True)[:n]



        def report_top_terms(self, texts,n , **kwargs):
            """
            Gives a dataframe with top terms and info on features
            Input:
                - n: number of top terms to output
                - texts: List of texts to extract the topterms 
                - **kwargs: different feature to have info on top terms -> category, gender, organization, sentiment etc...
            
            Output:
                - pd.Dataframe   
            """     
            report = pd.DataFrame()
            if self.top_terms=={}:
                top_terms=self.compute_top_terms(n=n,texts=texts)
            else:
                top_terms = self.top_terms
            report['top_terms'] = top_terms.keys()
            report['freq top_terms'] = top_terms.values()
            for feature in kwargs.keys():
                features_dict = [top_category(term, texts, kwargs[feature]) for term in top_terms]
                report[str(feature)] = [list(feature_element.keys())[0] for feature_element in features_dict]
                report['proba'] = [list(feature_element.values())[0] for feature_element in features_dict]
            return report



        def get_words_freq_by_category(self, texts, categories, n=100):
            """Gives a dict with categories and the top words for each category"""
            texts = np.array(texts)
            categories = np.array(categories)
            
            unique_categories = np.unique(categories)
            
            freq_categories = {}
            
            for cat in unique_categories:
                texts_cat = texts[categories == cat]
                freq_categories[cat] = dict(sorted(self.get_words_freqs(texts_cat, cat=cat).items(), key=operator.itemgetter(1),reverse=True)[:n])
                
            probas = {cat: 
                      {word: value.get(word) * len(texts[categories == cat]) \
                       / np.sum([freq_categories.get(_cat).get(word, 0) * len(texts[categories == _cat]) 
                                                            for _cat in unique_categories])
                      for word in value.keys()} 
                      for cat, value in freq_categories.items()}
            
            self.categories=probas
            return probas

                


def lemma_dict(top_terms, nlp):
    """A dict with lemmas"""
    word_list = [word for word in top_terms.keys()]
    lemmas = [lemmatizer(word, nlp) for word in word_list]
    dict_lemmas = {}
    for word, lemma in zip(word_list, lemmas):
        if lemma == '':
            continue
        if lemma not in dict_lemmas:
            dict_lemmas[lemma]= top_terms[word]
        else:
            dict_lemmas[lemma] += top_terms[word]
    return dict_lemmas




def lemmatizer(text, nlp):
    """Construct a lemmatizer"""
    doc = nlp(text)
    tokens = ' '.join([tok.lemma_ for tok in doc if tok.pos_ not in ['PUNCT', 'SPACE'] and len(tok.text) > 2])
    return tokens

def top_category(word, texts, categories):
    """Return the most common category for a word that appears on different categories"""
    category_word = np.array([category for text, category in zip(texts, categories) if word in text.lower()])
    most_common = Counter(category_word).most_common(1)
    return {most_common[0][0]: most_common[0][1]/category_word.shape[0]}