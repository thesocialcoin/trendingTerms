# Investigation

## Iteration 2 - Adding Stopwords

### Motivation

We detected in signals that some bigrams detected as trends contained stopwords. 
In the past we saw that stopwords were not usually detected as trends since their frequency is stable through time, that is why we did not decide to add stopwords in the code.
However we changed some parameters for the bigrams and observed that some stopwords were detected as trends.
In the example below we detect some trends like: *excusa* and since *excusa* is usually followed by *para* then we also have the trend *excusa para*.
To avoid this we decided to add some stopwords to the algorithm.

![Stopwords image](img/image_stopwords.png "Stopwords added to the algorithm")

### Fix

We downloaded a list of stopwords from the repository [python-stop-words](https://github.com/Alir3z4/python-stop-words. The stopwords were added in the [`stopwords.txt`](ds_trends/stop_words.txt) file.

Then since we use the [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) method from [scikit learn](https://scikit-learn.org/stable/index.html) we directly passed the list of stopwords to the method.

This is a quick fix that should be revised in the future since the stopwords are not the same in all the languages and we should be able to detect the language of the text and use the corresponding stopwords.
However to do so in proper way we should have a separated library for preprocessing that given a text, return the cleaned version of the text and without stopwords.

### Conclusions & Next Steps

**Conclusions**

- We added a list of stopwords to the algorithm to avoid detecting stopwords as trends.
The stopwords added are for the following languages: Arabic
Bulgarian
Catalan
Czech
Danish
Dutch
English
Finnish
French
German
Hungarian
Indonesian
Italian
Norwegian
Polish
Portuguese
Romanian
Russian
Spanish
Swedish
Turkish
Ukrainian


**Next Steps**

- Create a preprocessing library that given a text, return the cleaned version of the text and without stopwords.
- Once the library is created, eliminate the stopwords from this library


## Iteration 1 - Creating the library

The objective of the iteration was to create a library to allow us to detect unigram and bigram trends in a set of texts.

At Citibeats we use the algorithm for detecting trends at different stages like for data analysis, creation of reports and also for the product itself (in Signals), therefore we decided to create a library that could be used in all these stages.

**What is a trend?**

> A ***trend*** simply reflects what seems to be going around at any given time. A *trend* can be in any area and doesn't only reflect fashion, pop culture and entertainment. There can also be a *trend* in the stock market to be bullish or bearish, depending on economic indicators, or a political trend reflecting a nation’s current mood. Some trends are fun, some fabulous, some appalling, but however long they last, you can be sure there will always be a new *trend* coming along to replace the old. At Citibeats we study what are the trends in social media. Since we collect social media data we are able to detect what are the emerging expressions and have insights about what is happening in a country.

During this iteration we created the basic structure of the project, we created the main classes and the basic methods that we will need to implement the trending terms algorithm. We also created the first tests to check that the methods worked correctly.

**How do we detect trends?**

```
Input:
 - list of text in the past
 - list of text in the present


Output: 
 - Dictionnary with the trends of the present and their increase in frequency
```

The idea is given a list of texts that have been retrieved in the present and comparing them with the texts in the past to extract what are the words that are much more frequent in the present than in the past.

To do so we proceed with the following steps:

1. **Prepare Frequency Dictionaries**:
    - **1.1.** For the present texts:
        - Generate a dictionary with words and their frequencies.
        - Name this dictionary **`Dict_freqs_present`**.
    - **1.2.** For the past texts:
        - Similarly, generate a dictionary of words and their frequencies.
        - Name this dictionary **`Dict_freqs_past`**.
2. **Analyze Trends**:
    - 2.0 Create dictionary `**Dict_trends**`
    - **2.1.** Iterate over each word in **`Dict_freqs_present`**.
    - **2.2.** For the current word:
        - If its frequency is less than 10%, skip to the next word.
        - If the word exists in **`Dict_freqs_past`**:
            - Compare the word's frequency in present and past texts.
            - If the frequency in the present is significantly higher, mark the word as a trend.
            - Save its frequency as: **`Dict_trends[word]=(Dict_freqs_present[word] - Dict_freqs_past[word])/Dict_freqs_past[word]`**
        - If the word does not exist in **`Dict_freqs_past`**:
            - Check if the word appears in more than 20% of the present texts.
            - If it does, mark the word as a trend
            - Save its frequency as `**Dict_trends[word]**=9999`


### Next Steps

- Manage better **bigrams:** for example if “human rights” is a trend then “human” appears too as a trend but almost all the contribution was made by “human rights”
- Add more **stop words**