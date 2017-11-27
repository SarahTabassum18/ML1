# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 21:09:28 2017

@author: Lenovo
"""

import csv
 
typeF = []
text = []
 
with open('TrainFraudMail.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        typeF.append(row[0])
        text.append(row[2])
 
#print(typeF)
#print(text)


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(text)
print(X_train_counts.shape)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, typeF)

from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
              ('tfidf', TfidfTransformer()),
              ('clf', MultinomialNB()),
            ])
text_clf = text_clf.fit(text,typeF)

import numpy as np

typeF1 = []
text1 = []
 
with open('TestFraudMail.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        typeF1.append(row[0])
        text1.append(row[2])
        
predicted = text_clf.predict(text1)
print(np.mean(predicted == typeF1))
