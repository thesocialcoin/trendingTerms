# trendingTerms

Welcome to ds_trends, your go-to Python library for identifying trending words in large datasets!

In our increasingly digital age, understanding the buzzwords and trending topics is paramount for data analysts, marketers, journalists, and many more. Whether you're looking to spot the next big hashtag, analyze content dynamics, or just keep a finger on the pulse of textual data, ds_trends is here to help.

## Installation

In the CLI

```
pip install git+https://github.com/thesocialcoin/trendingTerms.git
```

## Usage Top Terms

First Load the libraries

```python
from ds_trends import top_terms_extractor
import pandas as pd
```
If you have a list of texts or a DataFrame with a column of texts then you can calculate the most frequent terms in the texts as follows:


```python
texts = data.loc[:, 'text'].values.tolist()
top_terms_algo = top_terms_extractor.top_terms_extractor()

terms = top_terms_algo.compute_top_terms(texts=texts, n=10)
for term, freq in terms.items():
    print(f'{term}: {freq*100:.2f}%')
```

## Usage Trending Terms


Load the libraries

```python
from ds_trends import trends
import pandas as pd
```

If you have a list of texts and dates or a DataFrame with a column of texts and another one of dates then you can extract the trends from one period wrt another as follows:

```python
texts_present = data.loc[data['date'] >= '2023-06-01', 'text'].values.tolist()
texts_past = data.loc[data['date'] < '2023-06-01', 'text'].values.tolist()

trends_algo = trends.TopTrends()
trends_words = trends_algo.get_trends(texts_present, texts_past, rate=0.6)

for word, freq in trends_words.items():
    print(f'{word}: {freq*100:.2f}%')
```