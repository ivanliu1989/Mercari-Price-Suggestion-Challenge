import nltk
import string
import re
import numpy as np
import pandas as pd
import pickle
# import lda

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white")

from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words

from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import plotly.offline as py

py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
# %matplotlib inline

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, output_notebook
# from bokeh.transform import factor_cmap

import warnings

warnings.filterwarnings('ignore')
import logging

logging.getLogger("lda").setLevel(logging.WARNING)
# Read in data set
train = pd.read_table("./data/train.tsv")
test = pd.read_table("./data/test.tsv")
train.dtypes
train.price.describe()

# Dist of target var
plt.subplot(1, 2, 1)
(train['price']).plot.hist(bins=50, figsize=(20, 10), edgecolor='white', range=[0, 250])
plt.xlabel('price+', fontsize=17)
plt.ylabel('frequency', fontsize=17)
plt.tick_params(labelsize=15)
plt.title('Price Distribution - Training Set', fontsize=17)

plt.subplot(1, 2, 2)
np.log(train['price'] + 1).plot.hist(bins=50, figsize=(20, 10), edgecolor='white')
plt.xlabel('log(price+1)', fontsize=17)
plt.ylabel('frequency', fontsize=17)
plt.tick_params(labelsize=15)
plt.title('Log(Price) Distribution - Training Set', fontsize=17)
plt.show()

# Shipping
train.shipping.value_counts() / len(train)
prc_shipBySeller = train.loc[train.shipping == 0, 'price']
prc_shipByBuyer = train.loc[train.shipping == 1, 'price']
fig, ax = plt.subplots(figsize=(20, 10))
ax.hist(np.log(prc_shipBySeller + 1), color='#8CB4E1', alpha=1.0, bins=50,
        label='Price when Seller pays Shipping')
ax.hist(np.log(prc_shipByBuyer + 1), color='#007D00', alpha=0.7, bins=50,
        label='Price when Buyer pays Shipping')
ax.set(title='Histogram Comparison', ylabel='% of Dataset in Bin')
plt.xlabel('log(price+1)', fontsize=17)
plt.ylabel('frequency', fontsize=17)
plt.title('Price Distribution by Shipping Type', fontsize=17)
plt.tick_params(labelsize=15)
plt.show()

# TOP 5 RAW CATEGORIES
train['category_name'].value_counts()[:5]


# reference: BuryBuryZymon at https://www.kaggle.com/maheshdadhich/i-will-sell-everything-for-free-0-55
def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")


train['general_cat'], train['subcat_1'], train['subcat_2'] = \
    zip(*train['category_name'].apply(lambda x: split_cat(x)))
train.head()
# repeat the same step for the test set
test['general_cat'], test['subcat_1'], test['subcat_2'] = \
    zip(*test['category_name'].apply(lambda x: split_cat(x)))

# 7 Main categories
x = train['general_cat'].value_counts().index.values.astype('str')
y = train['general_cat'].value_counts().values
pct = [("%.2f" % (v * 100)) + "%" for v in (y / len(train))]
trace1 = go.Bar(x=x, y=y, text=pct)
layout = dict(title='Number of Items by Main Category',
              yaxis=dict(title='Count'),
              xaxis=dict(title='Category'))
fig = dict(data=[trace1], layout=layout)
py.iplot(fig)

x = train['subcat_1'].value_counts().index.values.astype('str')[:15]
y = train['subcat_2'].value_counts().values[:15]
pct = [("%.2f" % (v * 100)) + "%" for v in (y / len(train))][:15]
trace1 = go.Bar(x=x, y=y, text=pct,
                marker=dict(
                    color=y, colorscale='Portland', showscale=True,
                    reversescale=False
                ))
layout = dict(title='Number of Items by SubCategory (Top 15)',
              yaxis=dict(title='Count'),
              xaxis=dict(title='SubCategory'))
fig = dict(data=[trace1], layout=layout)
py.iplot(fig)

general_cats = train['general_cat'].unique()
x = [train.loc[train['general_cat'] == cat, 'price'] for cat in general_cats]
data = [go.Box(x=np.log(x[i] + 1), name=general_cats[i]) for i in range(len(general_cats))]
layout = dict(title="Price Distribution by General Category",
              yaxis=dict(title='Frequency'),
              xaxis=dict(title='Category'))
fig = dict(data=data, layout=layout)
py.iplot(fig)

# Brand Name
x = train['brand_name'].value_counts().index.values.astype('str')[:10]
y = train['brand_name'].value_counts().values[:10]
trace1 = go.Bar(x=x, y=y,
                marker=dict(
                    color=y, colorscale='Portland', showscale=True,
                    reversescale=False
                ))
layout = dict(title='Top 10 Brand by Number of Items',
              yaxis=dict(title='Brand Name'),
              xaxis=dict(title='Count'))
fig = dict(data=[trace1], layout=layout)
py.iplot(fig)


# Item Description
def wordCount(text):
    # convert to lower case and strip regex
    try:
        # convert to lower case and strip regex
        text = text.lower()
        regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        txt = regex.sub(" ", text)
        # tokenize
        # words = nltk.word_tokenize(clean_txt)
        # remove words in stop words
        words = [w for w in txt.split(" ") \
                 if not w in stop_words.ENGLISH_STOP_WORDS and len(w) > 3]
        return len(words)
    except:
        return 0


# add a column of word counts to both the training and test set
train['desc_len'] = train['item_description'].apply(lambda x: wordCount(x))
test['desc_len'] = test['item_description'].apply(lambda x: wordCount(x))
df = train.groupby('desc_len')['price'].mean().reset_index()
trace1 = go.Scatter(
    x=df['desc_len'],
    y=np.log(df['price'] + 1),
    mode='lines+markers',
    name='lines+markers'
)
layout = dict(title='Average Log(Price) by Description Length',
              yaxis=dict(title='Average Log(Price)'),
              xaxis=dict(title='Description Length'))
fig = dict(data=[trace1], layout=layout)
py.iplot(fig)

train.item_description.isnull().sum()
# remove missing values in item description
train = train[pd.notnull(train['item_description'])]

# create a dictionary of words for each category
cat_desc = dict()
for cat in general_cats:
    text = " ".join(train.loc[train['general_cat'] == cat, 'item_description'].values)
    cat_desc[cat] = tokenize(text)

# flat list of all words combined
flat_lst = [item for sublist in list(cat_desc.values()) for item in sublist]
allWordsCount = Counter(flat_lst)
all_top10 = allWordsCount.most_common(20)
x = [w[0] for w in all_top10]
y = [w[1] for w in all_top10]

trace1 = go.Bar(x=x, y=y, text=pct)
layout = dict(title='Word Frequency',
              yaxis=dict(title='Count'),
              xaxis=dict(title='Word'))
fig = dict(data=[trace1], layout=layout)
py.iplot(fig)

# Text Mining
# Tokenization

# tf-idf

# t-SNE

# K-means

# LDA
