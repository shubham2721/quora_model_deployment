import numpy as np
import pandas as pd
from subprocess import check_output
import os
import gc

import re
from nltk.corpus import stopwords
import distance
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from thefuzz import fuzz
from sklearn.preprocessing import MinMaxScaler
from wordcloud import WordCloud, STOPWORDS
from os import path
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import pickle
import nltk
nltk.download('stopwords')

freq_qid1 = pd.read_csv('trained_model/qid1_frequency.csv')
freq_qid2 = pd.read_csv('trained_model/qid2_frequency.csv')

def pre_process(data):
#This function will make the string lower, removes HTML tag, link, date and also performs stemming. Basic string operations
    x = str(data).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will")

    # Replacing Date with date string
    pattern = '([0-2][0-9]|(3)[0-1])(\/|\.)(((0)[0-9])|((1)[0-2]))(\/|\.)\d{2,4}'
    x = re.sub(pattern, 'date', x)

    # Replacing links with link string
    pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    x = re.sub(pattern, 'link', x)

    #million and thousand representation
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)

    # Stemming
    porter = PorterStemmer()
    # Removing special chars
    pattern = re.compile('\W')

    if type(x) == type(''):
        x = re.sub(pattern, ' ', x)

    if type(x) == type(''):
        x = porter.stem(x)
        example1 = BeautifulSoup(x, features="html.parser")
        x = example1.get_text()

    return x


# basic feature extraction like frequency word common, length of strings etc
def normalized_word_Common(row):
        w1 = set(map(lambda word: word.lower().strip(), str(row['question1']).split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), str(row['question2']).split(" ")))    
        return 1.0 * len(w1 & w2)

def normalized_word_Total(row):
        w1 = set(map(lambda word: word.lower().strip(), str(row['question1']).split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), str(row['question2']).split(" ")))    
        return 1.0 * (len(w1) + len(w2))

def normalized_word_share(row):
        w1 = set(map(lambda word: word.lower().strip(), str(row['question1']).split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), str(row['question2']).split(" ")))    
        return 1.0 * len(w1 & w2)/(len(w1) + len(w2))

def safe_divide(numerator, denominator, default=0):
    if denominator == 0:
        return default
    return 1.0 * (numerator / denominator)

def get_basic_feature(df):

    df = pd.merge(df,freq_qid1, left_on = 'qid1', right_on = 'qid1_n' ,how = 'left') 
    df = pd.merge(df,freq_qid2, left_on = 'qid2', right_on = 'qid2_n' ,how = 'left') 

    df['freq_qid1'] = df['freq_qid1'].fillna(1)
    df['freq_qid2'] = df['freq_qid2'].fillna(1)

    df.drop(columns =['qid1_n', 'qid2_n'], inplace = True)

    df['q1len'] = df['question1'].str.len() 
    df['q2len'] = df['question2'].str.len()
    df['q1_n_words'] = df['question1'].apply(lambda row: len(str(row).split(" ")))
    df['q2_n_words'] = df['question2'].apply(lambda row: len(str(row).split(" ")))
    
    df['word_Common'] = df.apply(normalized_word_Common, axis=1)
    
    df['word_Total'] = df.apply(normalized_word_Total, axis=1)
    
    df['word_share'] = df.apply(normalized_word_share, axis=1)

    df['freq_q1+q2'] = df['freq_qid1']+df['freq_qid2']
    df['freq_q1-q2'] = abs(df['freq_qid1']-df['freq_qid2'])

    return df


# The NLP and Fuzzy Feature Extraction
def get_token_features(q1,q2):
    
    STOP_WORDS = stopwords.words("english")

    # creating list token_deatures
    token_features = [0.0]*10

    # token
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    q1_stopwords = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stopwords = set([word for word in q2_tokens if word in STOP_WORDS])
    
    # Feature Extraction NLP

    #1. CWC_MIN
    token_features[0] = safe_divide(len(q1_words & q2_words), min (len(q1_words), len(q2_words)))
    #2. CWC_MAX
    token_features[1] = safe_divide(len(q1_words & q2_words), max (len(q1_words), len(q2_words)))
    #3. CSC_MIN
    token_features[2] = safe_divide(len(q1_stopwords & q2_stopwords), min (len(q1_stopwords), len(q2_stopwords)))
    #4. CSC_MAX
    token_features[3] = safe_divide(len(q1_stopwords & q2_stopwords), max (len(q1_stopwords), len(q2_stopwords)))
    #5. CTC_MIN
    token_features[4] = safe_divide(len(set(q1_tokens) & set(q2_tokens)), min (len(q1_tokens), len(q2_tokens)))
    #6. CTC_MAX
    token_features[5] = safe_divide(len(set(q1_tokens) & set(q2_tokens)),max (len(q1_tokens), len(q2_tokens)))
    #7. last_word_eq
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    #8. first_word_eq
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    #9. abs_len_diff
    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    #10. mean_len
    token_features[9] = (len(q1_tokens) + len(q2_tokens))/2

    return token_features

# get the Longest Common sub string

def get_longest_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)

# Creating Columns for each features and also including some advanced fuzzy features
def feature_extraction(df):
    df["question1"] = df["question1"].fillna("").apply(pre_process)
    df["question2"] = df["question2"].fillna("").apply(pre_process)

    # Applying Feature

    # Merging Features with dataset
    
    token_features = df.apply(lambda x: get_token_features(x["question1"], x["question2"]), axis=1)

    df["cwc_min"]       = list(map(lambda x: x[0], token_features))
    df["cwc_max"]       = list(map(lambda x: x[1], token_features))
    df["csc_min"]       = list(map(lambda x: x[2], token_features))
    df["csc_max"]       = list(map(lambda x: x[3], token_features))
    df["ctc_min"]       = list(map(lambda x: x[4], token_features))
    df["ctc_max"]       = list(map(lambda x: x[5], token_features))
    df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
    df["first_word_eq"] = list(map(lambda x: x[7], token_features))
    df["abs_len_diff"]  = list(map(lambda x: x[8], token_features))
    df["mean_len"]      = list(map(lambda x: x[9], token_features))

    # Feature Extraction Fuzzy
    
    df["token_set_ratio"]    = df.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
    # The token sort approach involves tokenizing the string in question, sorting the tokens alphabetically, and 
    # then joining them back into a string We then compare the transformed strings with a simple ratio().
    df["token_sort_ratio"]   = df.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
    df["fuzz_ratio"]         = df.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
    df["fuzz_partial_ratio"] = df.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
    #df["longest_substr_ratio"]  = df.apply(lambda x: get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)
    return df

# '''TF_IDF feature extraction'''

# def tf_idf_w2v(df):
#     df['question1'] = df['question1'].apply(pre_process)
#     df['question2'] = df['question2'].apply(pre_process)

#     nlp = spacy.load("en_core_web_lg")
#     tf_idf = pickle.load(open('trained_model/tf_idf.pkl', 'rb'))
#     word2tfidf = dict(zip(tf_idf.get_feature_names_out(), tf_idf.idf_))
#     q1_w2v = []
#     q2_w2v = []

#     # for question1
#     for sentence in df['question1'].values:
#         word_vec1 =  nlp(sentence)
#         # for each word we need to have empty vectors of the dimension (1, number of dimension trained)
#         word_vec = np.zeros(word_vec1.vector.shape)
#         word_cnt = 0
#         # Step:2 Looping through each sentence words
#         for word in word_vec1:
#             # Obtaining the polarity and numerical vector for each word
#             vec = word.vector
#             if str(word) in word2tfidf.keys():
#                 # Step:3 Multiplying the numerical vector with its IDF
#                 word_vec += (word2tfidf[str(word)] * vec)
#             if np.sum(word_vec) != 0:
#                 word_cnt += 1
#         # Storing Avg IDF W2V for each sentences
#         q1_w2v.append(word_vec / word_cnt)

#     #for question2
#     for sentence in df['question2'].values:
#         word_vec1 =  nlp(sentence)
#         # for each word we need to have empty vectors of the dimension (1, number of dimension trained)
#         word_vec = np.zeros(word_vec1.vector.shape)
#         word_cnt = 0
#         # Step:2 Looping through each sentence words
#         for word in word_vec1:
#             # Obtaining the polarity and numerical vector for each word
#             vec = word.vector
#             if str(word) in word2tfidf.keys():
#                 # Step:3 Multiplying the numerical vector with its IDF
#                 word_vec += (word2tfidf[str(word)] * vec)
#             if np.sum(word_vec) != 0:
#                 word_cnt += 1
#         # Storing Avg IDF W2V for each sentences
#         q2_w2v.append(word_vec / word_cnt)

#     return q1_w2v, q2_w2v
