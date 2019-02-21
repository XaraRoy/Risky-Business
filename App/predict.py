from model import *
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import figure
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from operator import itemgetter
import collections
import seaborn as sns
from pathlib import Path
import colorsys
import colorama

from sklearn.externals import joblib


def estimator_load_model(model_time):
    model = joblib.load(f'../Data/Outputs/s2s{model_time}.pkl')
    vectorizer = joblib.load(f'../Data/Outputs/vec{model_time}.pkl')
    return model, vectorizer


def estimator_predict_string(string):
    empty_list = []
    print('Input String: %s' % string)
    print('\n')
    print('Prediction:')

    X = vectorizer.transform([string])
    predicted = model.predict(X)
    print('kmeans prediction: %s' % predicted)
    print("closest cluster centers :")
    for ind in order_centroids[predicted[0], :5]:
        print(' %s' % terms[ind])
    return X


def estimator_predict_document(document, name):
    dictionary_list = []
    for counter, sentence in enumerate(document.split(".")):
        if len(sentence) != 0:
            vector_matrix = vectorizer.transform([sentence])
            predicted_label = model.predict(vector_matrix)
            sentence_len = len(sentence.split(" "))
            sentence_info = {'company': name, 'sentence#': counter, 'text': sentence,
                             'wordcount': sentence_len, 'label': predicted_label[0]}
            dictionary_list.append(sentence_info)
    dataframe = pd.DataFrame(dictionary_list)
    dataframe["% of total"] = dataframe['wordcount'] /         sum(dataframe['wordcount'])
#         (name, sentence, predicted_label)
    return(dataframe)


def frame_it(doclist, names):
    frames = []
    for document, name in zip(doclist, names):
        frame = estimator_predict_document(document, name)
        frames.append(frame)

    muliple_company_frame = pd.concat(frames)
    muliple_company_frame.head()
    grouped_frame = muliple_company_frame.groupby(
        ['company', 'label']).agg({'% of total': 'sum'}).reset_index()
    return grouped_frame, muliple_company_frame


def prep_for_heatmap(muliple_company_frame):
    company_clusters = muliple_company_frame.groupby(['label', 'company']).agg(
        {'% of total': 'sum'}).unstack(level='company').fillna(0).T

    company_clusters = company_clusters.reset_index(level=0, drop=True)
    return company_clusters


def plot_heatmap(company_clusters):
    fig2, ax2 = plt.subplots(figsize=(20, 20))
    cmap = sns.light_palette('blue', as_cmap=True)

    sns.heatmap(company_clusters, ax=ax2, cmap=cmap)

    ax2.set_xlabel('Label', fontdict={'weight': 'bold', 'size': 14})
    ax2.set_ylabel('Company', fontdict={'weight': 'bold', 'size': 14})
    for label in ax2.get_xticklabels():
        label.set_size(16)
        label.set_weight("bold")
    for label in ax2.get_yticklabels():
        label.set_size(16)
        label.set_weight("bold")
    plt.savefig("../Data/Outputs/Heatmap.jpg", dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)



def get_N_HexCol(N):
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out




model, vectorizer = estimator_load_model('Feb14-0130PM')
print("Model LOADED!!")
doclist, names = make_pipeline(0, istesting=True)
print("Doclist Generated")
grouped_frame, muliple_company_frame = frame_it(doclist, names)
company_clusters = prep_for_heatmap(muliple_company_frame)
print("Dataframe Generated")
company_clusters.to_csv(path_or_buf='../Data/Outputs/WHATTONAMETHIS.csv')
plot_heatmap(company_clusters)
print("Heatmap Gnerated")

truek = 35
colormap = get_N_HexCol(truek)

for label in range(truek):
    color = colormap[label]

for counter, _ in enumerate(tqdm(doclist)):
    company = muliple_company_frame['company'].unique()[counter]
    companyFrame = muliple_company_frame[muliple_company_frame['company'] == company]
    for text, label in zip(companyFrame['text'], companyFrame['label']):
        color = colormap[label]
        display(Markdown(f'<font color="{color}">' +
                        text + f'  ({label})' + '</font>'))



print("Text Ready for Coloring")