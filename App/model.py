from __future__ import print_function
from nltk.stem.snowball import SnowballStemmer
import string
from nltk.tag import pos_tag
from gensim import corpora, models, similarities
from sklearn.externals import joblib
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity

import colorama
import heapq
import hdbscan
import seaborn as sns
from scipy.sparse import coo_matrix
from sklearn.metrics import silhouette_samples, silhouette_score
from pathlib import Path
import sys
import collections
from operator import itemgetter
import time
from tqdm import tqdm
import re

from datetime import datetime
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk import pos_tag
import nltk
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import figure
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics

import numpy as np
import pandas as pd
import matplotlib


def make_pipeline(NUMBER_OF_DOCS, istesting):
    doclist = []
    names = []
    Path_Input = input(
        "Enter a relative path, or hit enter for ../Data/Inputs:   ")
    if Path_Input == "":
        pathlist = Path("../Data/Inputs").glob('**/*.txt')
    else:
        pathlist = Path(Path_Input).glob('**/*.txt')

    try:
        for path in tqdm(pathlist):
            path_in_str = str(path)
            name = path_in_str.split("\\")[3].split(".")[0]
            names.append(name.replace("_", " "))
            # TODO SPLIT PATH TO COMPANY NAME, make Index
            file = open(path, "r")
            # print "Output of Readlines after appending"
            text = file.readlines()
        #     print(text[:10])
            doclist.append(text[0])

    except IndexError as I:
        print(I, "Did you enter a correct path?")
        sys.exit()
    except ValueError as I:
        print(I, "Did you enter a correct path?")
        sys.exit()
    if istesting is True:
        print('Test set generated')
        doclist = doclist[NUMBER_OF_DOCS:NUMBER_OF_DOCS * 2]
        names = names[NUMBER_OF_DOCS: NUMBER_OF_DOCS * 2]
    if len(doclist) > NUMBER_OF_DOCS and NUMBER_OF_DOCS != -1:
        doclist = doclist[:NUMBER_OF_DOCS]
        names = names[:NUMBER_OF_DOCS]

    print('%s docs loaded' % len(names))
    print()
    if len(names) > 5:
        print(names[:5])
    if len(names) > 10:
        print('......', names[-5:])
    return doclist, names


def transform_tokens(doclist):
    token_list = []
    for doc in tqdm(doclist, desc="Tokenizing", leave=False):
        dirty_tokens = nltk.sent_tokenize(doc)
        token_list += [dirty_tokens]
    return token_list


def transform_filtered(token_list, doclist, names):
    punc = ['.', ',', '"', "'", '?', '!', ':',
            ';', '(', ')', '[', ']', '{', '}', "%"]
    more_stops = ['\\t\\t\\t',
                  '\\t\\t\\', '\\t\\t\\t',
                  '<U+25CF>', '[1]', 'feff', '1a', 'item']
    maybe_bad_stops = ['may', 'could',  'contents',
                       'table', 'time', '25cf', 'factors', 'risk',
                       ]
    global Stopwords_list
    Stopwords_list = stopwords.words(
        'english') + more_stops + punc + maybe_bad_stops
    filtered_tokens = []
    names_list = []
    if len(names) > 1:
        if len(token_list) != len(doclist):
            token_list = [token_list]
        index = 0

        for tokens in tqdm(token_list, desc="Filtering Documents", leave=True):
            filtered_docs = []
            name = names[index]
            for token in tqdm(tokens, desc="Filtering Words", leave=False):
                if re.search(r'\d{1,}', token):  # getting rid of digits
                    pass
                else:
                    #                 NNP proper noun, singular ‘Harrison’
                    #                 NNPS proper noun, plural ‘Americans’
                    if token not in Stopwords_list:
                        if pos_tag(token) != 'NNP' and pos_tag(
                          token) != 'NNPS':
                            filtered_docs.append(token.lower())
                        else:
                            filtered_docs.append('proper_noun')
                        names_list.append(name)
            index += 1
            filtered_tokens.append(filtered_docs)
        else:
            for token in tqdm(tokens, desc="Filtering Words", leave=False):
                if re.search(r'\d{1,}', token):  # getting rid of digits
                    pass
                else:
                    #                 NNP proper noun, singular ‘Harrison’
                    #                 NNPS proper noun, plural ‘Americans’
                    if token not in Stopwords_list:
                        if pos_tag(token) != 'NNP' and pos_tag(
                          token) != 'NNPS':
                            filtered_docs.append(token.lower())
                        else:
                            filtered_docs.append('proper_noun')
    return filtered_tokens, names_list


def transform_stemming(filtered_tokens):
    stemmed = []
    for doc in filtered_tokens:
        for token in doc:
            stemmed.append(PorterStemmer().stem(token))
            # stemmed.append(LancasterStemmer().stem(token))
            # stemmed.append(SnowballStemmer('english').stem(token))

    return stemmed


def transform_vectorize(stemmed, smallest_ngram, largest_ngram):

    vectorizer = TfidfVectorizer(stop_words=Stopwords_list,
                                 ngram_range=(
                                     smallest_ngram, largest_ngram),
                                 max_df=0.55, min_df=0.01)
#     vectorizer = CountVectorizer(stop_words=Stopwords_list,
#                                  ngram_range=(
#                                   smallest_ngram, largest_ngram),
#                                    max_df=0.75, min_df=0.01)
    sparseMatrix = vectorizer.fit_transform(stemmed)
    return sparseMatrix, vectorizer


def GridSearch():
    from sklearn.model_selection import GridSearchCV
    model = KMeans(init='k-means++', random_state=42, n_init=15)
    param_grid = {'max_iter': [10, 50, 100, 150, 200,
                               250, 300, 350, 400, 500, 1000],
                  'n_clusters': [25, 30, 33, 35]}
    grid = GridSearchCV(model, param_grid, verbose=3, n_jobs=8)
    grid.fit(sparseMatrix)
    lids = model.cluster_centers_
    score = model.score(sparseMatrix)
    silhouette_score = metrics.silhouette_score(
        sparseMatrix, labels, metric='euclidean')
    print(grid.best_params_)
    print(grid.best_score_)

    return grid, model, score, silhouette_score


def estimator_cluster(sparseMatrix, vectorizer):
    truek = 35
    model = KMeans(n_clusters=truek, init='k-means++',
                   max_iter=50, n_init=1, random_state=42,
                   )
    model.fit(sparseMatrix)
    model_time = datetime.now().strftime("%b%d-%I%M%p")
    joblib.dump(model,  f'../Data/Outputs/s2s{model_time}.pkl')
    joblib.dump(vectorizer,  f'../Data/Outputs/vec{model_time}.pkl')
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    print(f"Model Generated at {model_time}")
    labels = model.labels_
    centroids = model.cluster_centers_
    print("Cluster id labels for inputted data")
    print(labels)
    print("Centroids data")
    print(centroids)

    kmeans_score = model.score(sparseMatrix)
    print("Score (Opposite of the value of X on the K-means objective, \n",
          "which is Sum of distances of samples to their closest cluster",
          " center):")
    print(kmeans_score)

    silhouette_score = metrics.silhouette_score(
        sparseMatrix, labels, metric='euclidean')

    print("Silhouette_score: ")
    print(silhouette_score)

    return terms, order_centroids, model,
    truek, model_time, kmeans_score, silhouette_score


def generate_model():
    num_docs = int(input(
        "How many docs to process? (-1 for entire folder) : "))
    if num_docs < 2 and num_docs != -1:
        print('More documents please!')
        generate_model()
    doclist, names = make_pipeline(num_docs, istesting=False)
    tokens = transform_tokens(doclist)
    filtered_tokens, _ = transform_filtered(tokens, doclist, names)
    stemmed = transform_stemming(filtered_tokens)
    smallest_ngram = 1
    largest_ngram = len(max(tokens, key=len))
    sparseMatrix, vectorizer = transform_vectorize(
        stemmed, smallest_ngram, largest_ngram)
    estimator_cluster(sparseMatrix, vectorizer)

generate_model()
