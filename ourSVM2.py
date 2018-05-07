# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 21:09:28 2017

@author: Lenovo
"""

import csv
import numpy as np
#from sklearn import svm
from sklearn.cross_validation import train_test_split
#from sklearn.metrics import accuracy_score


X = []
#Z= []
#merge=[]
#subject=[]
y = []
 
with open('Fraud_Email_dataset.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        X.append(row[2])
        #Z.append(row[1])
        #merge= X + Z
        #subject.append(row[1])
        y.append(row[0])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
 
#print(typeF)
#print(text)
print(X_train[2])


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
print(X_train_counts.shape)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print(X_train_tfidf)
print(X_train_tfidf.shape)

from sklearn.svm import LinearSVC
clf = LinearSVC().fit(X_train_tfidf, y_train)

from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
              ('tfidf', TfidfTransformer()),
              ('clf', LinearSVC()),
            ])
text_clf = text_clf.fit(X_train,y_train)        
predicted = text_clf.predict(X_test)
print(np.mean(predicted == y_test))

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)
#svm_model = svm.SVC(kernel='linear', C=1, gamma='auto')
#svm_model.fit(X_train,y_train)
#predictions = svm_model.predict(X_test)
#accuracy_score(y_test,predicted)
#print(accuracy_score)
from sklearn.model_selection import cross_val_score
#clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(text_clf, X, y, cv=5)
print(scores)

print("Accuracy: %0.8f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted)

print(cm)


#plot confusion matrix
def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,normalize=True):
    import matplotlib.pyplot as plt
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
       cmap = plt.get_cmap('Blues')
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    if target_names is not None:
       tick_marks = np.arange(len(target_names))
       plt.xticks(tick_marks, target_names, rotation=45)
       plt.yticks(tick_marks, target_names)
    
    if normalize:
       cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
       if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
            horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
       else:
           plt.text(j, i, "{:,}".format(cm[i, j]),
           horizontalalignment="center",
           color="white" if cm[i, j] > thresh else "black")
    
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
plot_confusion_matrix(cm      ,
                      normalize    = False,
                      target_names = ['Fraud', 'Non-Fraud'],
                      title        = "Confusion Matrix")    

'''
#precision
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, predicted)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt



precision, recall, _ = precision_recall_curve(y_test, predicted)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
'''