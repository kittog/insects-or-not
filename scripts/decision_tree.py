# decision tree
import pandas as pd
import numpy as np
import os
import regex as re

from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

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

# prep data
X = res
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# X_train, y_train = make_blobs(n_samples=1000, n_features=100, centers=100, random_state=10)

clf = RandomForestClassifier(max_depth=None, min_samples_split=2, n_estimators=10)
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

scores = cross_val_score(clf, X, y, cv=5)
print(scores.mean())