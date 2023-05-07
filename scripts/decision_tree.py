# decision tree
import pandas as pd
import numpy as np
import os
import regex as re
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from nltk.corpus import stopwords

df = pd.read_csv("insects_or_not.csv")
df['text'] = df['text'].str.lower()
df['text'] = df['text'].str.replace("\n", "")
df['text'] = df['text'].str.replace("\[.*?\]", "")
df['text'] = df['text'].str.replace("({|})+", "")
df['text'] = df['text'].str.replace("https?://S+|www\.\S+", "")
# tfidf
stopwords_fr = stopwords.words('english')
vec = TfidfVectorizer()
x = vec.fit_transform(df['text'])
# x.shape = (221, 8906)

# svd (réduction de données)
svd_100 = TruncatedSVD(n_components=100)
svd_200 = TruncatedSVD(n_components=200)
res_100 = svd_100.fit_transform(x)
res_100 = svd_200.fit_transform(x)
ex_var_100 = svd_100.explained_variance_
ex_var_200 = svd_200.explained_variance_
#print(svd.explained_variance_)
# res.shape = (221, 100)

sum_var_100 = np.cumsum(ex_var_100)
sum_var_200 = np.cumsum(ex_var_200)

plt.plot(sum_var_200)
plt.xlabel("n_components")
plt.ylabel("variance cummulative sum")
plt.legend()
plt.title("Somme cumulative de la variance des données post SVD")
plt.savefig("cum_sum_svd_200.png")

plt.show()

# prep data
X = res_100
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


clf = LinearSVC(C=1.0, kernel="rbf", degree=3, tol=0.001, max_iter=-1, random_state=42)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


