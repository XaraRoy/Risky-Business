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
from markdown2 import markdown as marked
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import jinja2
import markdown
import os

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
    for document, name in tqdm(zip(doclist, names), desc="Predicting Documents"):
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
doclist, names = make_pipeline(1000, istesting=True)
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

# Reference
# https://gist.github.com/jiffyclub/5015986


TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <link href="http://netdna.bootstrapcdn.com/twitter-bootstrap/2.3.0/css/bootstrap-combined.min.css" rel="stylesheet">
    <style>
        body {
            font-family: sans-serif;
        }
        code, pre {
            font-family: monospace;
        }
        h1 code,
        h2 code,
        h3 code,
        h4 code,
        h5 code,
        h6 code {
            font-size: inherit;
        }
    </style>
</head>
<body>
<div class="container">
{{content}}
</div>
</body>
</html>
"""

markdowns = []

for counter, _ in enumerate(tqdm(doclist, desc="Generating Color Text HTML")):
    company = muliple_company_frame['company'].unique()[counter]
    companyFrame = muliple_company_frame[muliple_company_frame['company'] == company]
    name = names[counter].replace(" ", "_")
    md_file = f"../Data/{name}.md"
    extensions = ['extra', 'smarty']

    with open(md_file, 'w') as f:
        f.write(marked(f'<h1>{name.replace("_"," ")}</h1>'))
        f.write(marked('<h3>Year</h3>'))
        for text, label in zip(companyFrame['text'], companyFrame['label']):
            color = colormap[label]
            f.write(marked(f'<font color="{color}">' +  text + f'  ({label})' + '</font>'))
    with open(md_file, 'r') as f:
        html = markdown.markdown(f.read(), extensions=extensions, output_format='html5')
        doc = jinja2.Template(TEMPLATE).render(content=html) 
    with open(f'../Data/Outputs/ColorTexts/{name}.html', 'w') as k:
        k.write(doc)

    os.remove(md_file)