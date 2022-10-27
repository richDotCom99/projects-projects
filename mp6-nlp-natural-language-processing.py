#!/usr/bin/env python
# coding: utf-8

# In[34]:


get_ipython().run_line_magic('logstop', '')
get_ipython().run_line_magic('logstart', '-rtq ~/.logs/nlp.py append')
import seaborn as sns
sns.set()


# In[7]:


from static_grader import grader


# In[8]:


from IPython.display import clear_output

import warnings
warnings.filterwarnings('ignore')


# # NLP Miniproject

# ## Introduction
# 
# The objective of this miniproject is to gain experience with natural language processing and how to use text data to train a machine learning model to make predictions. For the miniproject, we will be working with product review text from Amazon. The reviews are for only products in the "Electronics" category. The objective is to train a model to predict the rating, ranging from 1 to 5 stars.
# 
# ## Scoring
# 
# For most of the questions, you will be asked to submit the `predict` method of your trained model to the grader. The grader will use the passed `predict` method to evaluate how your model performs on a test set with respect to a reference model. The grader uses the [R<sup>2</sup>-score](https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score) for model evaluation. If your model performs better than the reference solution, then you can score higher than 1.0. For the last question, you will submit the results of an analysis and your passed answer will be compared directly to the reference solution.
# 
# ## Downloading and loading the data
# 
# The data set is available on Amazon S3 and comes as a compressed file where each line is a JSON object. To load the data set, we will need to use the `gzip` library to open the file and decode each JSON into a Python dictionary. In the end, we have a list of dictionaries, where each dictionary represents an observation.

# In[10]:


get_ipython().run_cell_magic('bash', '', 'mkdir data\nwget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_electronics_reviews_training.json.gz -nc -P ./data')


# In[11]:


import gzip
import ujson as json
from pandas.io.json import json_normalize

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]


# In[12]:


json_normalize(data[0])


# The ratings are stored in the keyword `"overall"`. You should create an array of the ratings for each review, preferably using list comprehensions.

# In[13]:


ratings = [review['overall'] for review in data]


# In[14]:


ratings[:5]


# **Note**, the test set used by the grader is in the same format as that of `data`, a list of dictionaries. Your trained model needs to accept data in the same format. Thus, you should use `Pipeline` when constructing your model so that all necessary transformation needed are encapsulated into a single estimator object.

# ## Question 1: Bag of words model
# 
# Construct a machine learning model trained on word counts using the bag of words algorithm. Remember, the bag of words is implemented with `CountVectorizer`. Some things you should consider:
# 
# * The reference solution uses a linear model and you should as well; use either `Ridge` or `SGDRegressor`.
# * The text review is stored in the key `"reviewText"`. You will need to construct a custom transformer to extract out the value of this key. It will be the first step in your pipeline.
# * Consider what hyperparameters you will need to tune for your model.
# * Subsampling the training data will boost training times, which will be helpful when determining the best hyperparameters to use. Note, your final model will perform best if it is trained on the full data set.
# * Including stop words may help with performance.

# In[15]:


from sklearn.base import BaseEstimator,TransformerMixin

class KeySelector (BaseEstimator,TransformerMixin):
    def __init__(self, key):
        self.key = key
    
    def fit(self, X, y = None):
        return self 
    
    def transform(self, X):
        return [row[self.key] for row in X]


# In[16]:


from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, SGDRegressor
from spacy.lang.en import STOP_WORDS


# In[73]:


bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', CountVectorizer()),
    ('regressor', Ridge())
])

bag_of_words_model.fit(data, ratings)


# In[74]:


grader.score.nlp__bag_of_words_model(bag_of_words_model.predict)


# ## Question 2: Normalized model
# 
# Using raw counts will not be as effective compared if we had normalized the counts. There are several ways to normalize raw counts; the `HashingVectorizer` class has the keyword `norm` and there is also the `TfidfTransformer` and `TfidfVectorizer` that perform tf-idf weighting on the counts. Apply normalization to your model to improve performance.

# In[17]:


normalized_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge())
])

normalized_model.fit(data, ratings)


# In[18]:


grader.score.nlp__normalized_model(normalized_model.predict)


# ## Question 3: Bigrams model
# 
# The model performance may increase when including additional features generated by counting bigrams. Include bigrams to your model. When using more features, the risk of overfitting increases. Make sure you try to minimize overfitting as much as possible.

# In[19]:


bigrams_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer(lowercase=True, ngram_range=(1,2))),
    ('regressor', Ridge())
])

bigrams_model.fit(data, ratings)


# In[21]:


grader.score.nlp__bigrams_model(bigrams_model.predict)


# ## Question 4: Polarity analysis
# 
# Let's derive some insight from our analysis. We want to determine the most polarizing words in the corpus of reviews. In other words, we want identify words that strongly signal a review is either positive or negative. For example, we understand a word like "terrible" will mostly appear in negative rather than positive reviews. The naive Bayes model calculates probabilities such as $P(\text{terrible } | \text{ negative})$, the probability the word "terrible" appears in the text, given that the review is negative. Using these probabilities, we can derive a **polarity score** for each counted word,
# 
# $$
# \text{polarity} =  \log\left(\frac{P(\text{word } | \text{ positive})}{P(\text{word } | \text{ negative})}\right).
# $$ 
# 
# The polarity analysis is an example where a simpler model offers more explicability than a more complicated model. For this question, you are asked to determine the top twenty-five words with the largest positive **and** largest negative polarity, for a total of fifty words. For this analysis, you should:
# 
# 1. Use the naive Bayes model, `MultinomialNB`.
# 1. Use tf-idf weighting.
# 1. Remove stop words.
# 
# A trained naive Bayes model stores the log of the probabilities in the attribute `feature_log_prob_`. It is a NumPy array of shape (number of classes, the number of features). You will need the mapping between feature index to word. For this problem, you will use a different data set; it has been processed to only include reviews with one and five stars. You can download it below.

# In[22]:


get_ipython().run_cell_magic('bash', '', 'wget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_one_and_five_star_reviews.json.gz -nc -P ./data')


# In order to avoid memory issues, let's delete the older data.

# In[23]:


del data, ratings


# In[24]:


import numpy as np
from sklearn.naive_bayes import MultinomialNB

with gzip.open("data/amazon_one_and_five_star_reviews.json.gz", "r") as f:
    data_polarity = [json.loads(line) for line in f]

ratings = [review['overall'] for review in data_polarity]


# In[25]:


pipe = Pipeline([
    ("selector", KeySelector("reviewText")),
    ("vectorizer", TfidfVectorizer(stop_words = STOP_WORDS)),
    ("classifier", MultinomialNB())
])

pipe.fit(data_polarity, ratings)


# In[47]:


#retrieve features' log probabilities
log_prob = pipe['classifier'].feature_log_prob_

#compute polarity
polarity = log_prob[0,:] - log_prob[1,:]

#get feature names
terms = pipe['vectorizer'].get_feature_names()

#assign each feature to corresponding polarity
terms_polarity = list(zip(polarity, terms))

#sort features_polarity elements with respect to polarity
sorted_terms_polarity = sorted(terms_polarity)

#select N highest polirized terms
N = 50
highest_polarized_terms = sorted_terms_polarity[:N//2] + sorted_terms_polarity[-N//2:]


#extract the terms with highest polarization
top_50 = [term for polarity, term in highest_polarized_terms]


# In[48]:


grader.score.nlp__most_polar(top_50)


# ## Question 5: Topic modeling [optional]
# 
# Topic modeling is the analysis of determining the key topics or themes in a corpus. With respect to machine learning, topic modeling is an unsupervised technique. One way to uncover the main topics in a corpus is to use [non-negative matrix factorization](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html). For this question, use non-negative matrix factorization to determine the top ten words for the first twenty topics. You should submit your answer as a list of lists. What topics exist in the reviews?

# In[ ]:


from sklearn.decomposition import NMF
 


# *Copyright &copy; 2021 WorldQuant University. This content is licensed solely for personal use. Redistribution or publication of this material is strictly prohibited.*
