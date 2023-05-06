import pandas as pd
import numpy as np
import os
import regex as re
import seaborn as sns
from matplotlib import rcParams

from nltk.tree import *
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


nltk.download('maxent_ne_chunker')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('words')


df = pd.read_csv("insects_or_not.csv")
df['text'] = df['text'].str.lower()
df['text'] = df['text'].str.replace("\n", "")
df['text'] = df['text'].str.replace("\[.*?\]", "")
df['text'] = df['text'].str.replace("({|})+", "")
df['text'] = df['text'].str.replace("https?://S+|www\.\S+", "")
# tfidf
vec = TfidfVectorizer()
x = vec.fit_transform(df['text'])
# x.shape = (221, 8906)

# svd (réduction de données)
svd = TruncatedSVD(n_components=100)
res = svd.fit_transform(x)
# res.shape = (221, 100)

# model
from sklearn import svm
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

y = df['label'].values
X = res

cross_validation = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

model = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo')

# train
metrics = cross_validate(model, res, y, scoring=['precision_macro', 'recall_macro'],
	cv=cross_validation, n_jobs=1)

p = np.mean(metrics['test_precision_macro'])
p_std = np.std(metrics["test_precision_macro"])

r = np.mean(metrics['test_precision_macro'])
r_std = np.std(metrics["test_recall_macro"])

# print results
print(f'''Precision: {p} ({p_std})''')
print(f'''Recall: {r} ({r_std})''')
