from __future__ import print_function
from nltk.stem.snowball import SnowballStemmer
import string
from nltk.tag import pos_tag
from gensim import corpora, models, similarities
from sklearn.externals import joblib
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity


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
from tqdm.auto import tqdm
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
    pathlist = input("Enter a relative path, or hit enter for '../Data/Inputs'")
    if pathlist == "":
        pathlist = Path("../Data/Inputs").glob('**/*.txt')
    for path in tqdm(pathlist):
        # because path is object not string
        path_in_str = str(path)
        name = path_in_str.split("\\")[3].split(".")[0]
        names.append(name.replace("_", " "))
        file = open(path, "r")
        text = file.readlines()
        doclist.append(text[0])
    if istesting == True:
        print('Test set generated')
        doclist = doclist[NUMBER_OF_DOCS:NUMBER_OF_DOCS * 2]
        names = names[NUMBER_OF_DOCS: NUMBER_OF_DOCS * 2]
    if len(doclist) > NUMBER_OF_DOCS:
        doclist = doclist[:NUMBER_OF_DOCS]
        names = names[:NUMBER_OF_DOCS]

    print('%s docs loaded' % len(names))
    print()
    print(names[:5], '......',  names[-5:])

    
    return doclist, names



def transform_tokens(doclist):
    token_list = []
    for doc in tqdm(doclist, desc="Tokenizing", leave=True):
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
    #WE CAN RUN 1 or many docs at once#
    names_list = []
    if len(names) > 1:
        if len(token_list) != len(doclist):
            token_list = [token_list]
        index = 0

        for tokens in tqdm(token_list, desc="Filtering Documents"):
            filtered_docs = []
            name = names[index]
            for token in tqdm(tokens, desc="Filtering Words", leave=False):
                if re.search(r'\d{1,}', token):  # getting rid of digits
                    pass
                else:
                    #                 NNP proper noun, singular ‘Harrison’
                    #                 NNPS proper noun, plural ‘Americans’
                    if token not in Stopwords_list:
                        if pos_tag(token) != 'NNP' and pos_tag(token) != 'NNPS':
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
                        if pos_tag(token) != 'NNP' and pos_tag(token) != 'NNPS':
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
                                 ngram_range=(smallest_ngram, largest_ngram), max_df=0.55, min_df=0.01)
#     vectorizer = CountVectorizer(stop_words=Stopwords_list,
#                                  ngram_range=(smallest_ngram, largest_ngram), max_df=0.75, min_df=0.01)
    sparseMatrix = vectorizer.fit_transform(stemmed)
    return sparseMatrix, vectorizer





def GridSearch():
    from sklearn.model_selection import GridSearchCV
    model = KMeans(init='k-means++', random_state=42, n_init=15
                       )
    param_grid = {'max_iter': [10, 50, 100, 150, 200, 250, 300, 350, 400, 500, 1000],
                 'n_clusters': [25,30, 33, 35],
                 }
    grid = GridSearchCV(model, param_grid, verbose=3, n_jobs=8)
    grid.fit(sparseMatrix)

    lids = model.cluster_centers_

    score = model.score(sparseMatrix)
    silhouette_score = metrics.silhouette_score(sparseMatrix, labels, metric='euclidean')
    print(grid.best_params_)
    print(grid.best_score_)

    return grid, model, score, silhouette_score




def estimator_cluster(sparseMatrix, vectorizer):
    truek = 35  # FROM GRID SEARCH
    model = KMeans(n_clusters=truek, init='k-means++',
                   max_iter=50, n_init=1, random_state=42,
                   )
    model.fit(sparseMatrix)

    model_time = datetime.now().strftime("%b%d-%I%M%p")
    joblib.dump(model,  f'../Data/Outputs/s2s{model_time}.pkl')
    joblib.dump(vectorizer,  f'../Data/Outputs/vec{model_time}.pkl')
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()

    return terms, order_centroids, model, truek, model_time





def estimator_ppscore(model):
    labels = model.labels_
    centroids = model.cluster_centers_

    print(f"Model Generated at {model_time}")

    print("Cluster id labels for inputted data")
    print(labels)
    print("Centroids data")
    #print (centroids)

    kmeans_score = model.score(sparseMatrix)
    print("Score (Opposite of the value of X on the K-means objective, \n",
          "which is Sum of distances of samples to their closest cluster center):")
    print(kmeans_score)

    silhouette_score = metrics.silhouette_score(
        sparseMatrix, labels, metric='euclidean')

    print("Silhouette_score: ")
    print(silhouette_score)
    return kmeans_score, silhouette_score









# # SUMMARIZATION OF CORPORATE RISK FACTOR DISCLOSURE THROUGH TOPIC MODELING by Bao, Datta
# strings = [
#     'Topic 0: investment, property, distribution, interest, agreement',
#     'Topic 1: regulation, change, law, financial, operation, tax, accounting ',
#     'Topic 2: gas, price, oil, natural, operation, production Input prices risks ',
#     'Topic 3: stock, price, share, market, future, dividend, security, stakeholder ',
#     'Topic 4: cost, regulation, environmental, law, operation, liability',
#     'Topic 5: control, financial, internal, loss, reporting, history ',
#     'Topic 6: financial, litigation, operation, condition, action, legal, liability, regulatory, claim, lawsuit'
#     'Topic 7: competitive, industry, competition, highly',
#     'Topic 8: cost, operation, labor, operating, employee, increase, acquisition ',
#     'Topic 9: product, candidate, development, approval, clinical, regulatory',
#     'Topic 10: tax, income, asset, net, goodwill, loss, distribution, impairment, intangible ',
#     'Topic 11: interest, director, officer, trust, combination, share, conflict ',
#     'Topic 12: product, liability, claim, market, insurance, sale, revenue Potential defects in products',
#     'Topic 13: loan, real, estate, investment, property, market, loss, portfolio ',
#     'Topic 14: personnel, key, retain, attract, management, employee ',
#     'Topic 15: stock, price, operating, stockholder, fluctuate, interest, volatile  ',
#     'Topic 16: acquisition, growth, future, operation, additional, capital, strategy ',
#     'Topic 17: condition, economic, financial, market, industry, change, affected, downturn, demand Macroeconomic risks ',
#     'Topic 18: system, service, information, failure, product, operation, software, network, breach, interruption Disruption of operations'
#     'Topic 19: cost, contract, operation, plan, increase, pension, delay',
#     'Topic 20: customer, product, revenue, sale, supplier, relationship, key, portion, contract, manufacturing, rely Rely on few large customers',
#     'Topic 21: property, intellectual, protect, proprietary, technology, patent, protection, harm',
#     'Topic 22: product, market, service, change, sale, demand, successfully, technology, competition Volatile demand and results',
#     'Topic 23: provision, law, control, change, stock, prevent, stockholder, Delaware, charter, delay, bylaw',
#     'Topic 24: regulation, government, change, revenue, contract, law, service',
#     'Topic 25: capital, credit, financial, market, cost, operation, rating, access, liquidity, downgrade ',
#     'Topic 26: debt, indebtedness, cash, obligation, financial, credit, ',
#     'Topic 27: operation, international, foreign, currency, rate, fluctuation',
#     'Topic 28: loss, insurance, financial, loan, reserve, operation, cover',
#     'Topic 29: operation, natural, facility, disaster, event, terrorist, weather ']
# topics = [topic.split(":")[1] for topic in strings]


# targets = {
#     "Shareholder’s interest risk": topics[0],
#     "Regulation changes(accounting)": topics[1],
#     "Stakeholder’s profit": topics[2],
#     "Regulation changes(environment)": topics[3],
#     "Legal Risks": topics[4],
#     "Financial condition risks ": topics[5],
#     " Potential/Ongoing Lawsuits": topics[6],
#     "market Competition risks": topics[7],
#     "**Labor cost ": topics[8],
#     " New product introduction risks ": topics[9],
#     "**Accounting,  +Restructuring risks ": topics[10],
#     "**Management": topics[11],
#     " Potential defects in products": topics[12],
#     "**Investment": topics[13],
#     "Human resource risks": topics[13],
#     "Volatile stock price risks": topics[14],
#     "Merger & Acquisition risks": topics[15],
#     " +Industry is cyclical": topics[16],
#     " **Postpone ":  topics[17],
#     " +Infrastructure risks": topics[18],
#     "+Suppliers risks +Downstream risks": topics[19],
#     "license Intellectual property risks": topics[20],
#     "+Licensing related risks' ": topics[21],
#     "+ Competition risks ": topics[22],
#     "*Potential/Ongoing Lawsuits*": topics[23],
#     "Regulation changes": topics[24],
#     "Credit risks": topics[25],
#     "covenant Funding risks ": topics[26],
#     "International risks": topics[27],
#     #     "Insurance" : topics[28],
#     #     "Catastrophes" : topics[29]
# }




# for topic in topics:
#     print(topic)
#     estimator_predict_string(topic)



num_docs = input("How many docs to process? : ")

doclist, names = make_pipeline(num_docs, istesting=False)
tokens = transform_tokens(doclist)
filtered_tokens, names_list = transform_filtered(tokens, doclist, names)
stemmed = transform_stemming(filtered_tokens)


largest_ngram = 15
smallest_ngram = 1
largest_ngram = len(max(tokens, key=len))
sparseMatrix, vectorizer = transform_vectorize(stemmed, smallest_ngram, largest_ngram)


terms, order_centroids, model, truek, model_time = estimator_cluster(sparseMatrix, vectorizer)
estimator_ppscore(model)
