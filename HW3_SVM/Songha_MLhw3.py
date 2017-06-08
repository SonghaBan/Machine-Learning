# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:37:30 2017

@author: song-isong-i
"""

from sklearn import svm
from sklearn.model_selection import cross_val_score

def main():
    clf = svm.SVC(kernel='rbf',gamma = 0.005, C=1.5)
    X, Y = normalize('mnist_train.txt')
    Xt, Yt = normalize('mnist_test.txt')
    E = clf.fit(X,Y)
    print(E)
    clf.predict(Xt)
    print("Error Rate :",test_error(Yt, clf.predict(Xt)),"%")
    scores = cross_val_score(E, X, Y, cv=5)
    print(scores)
    print("Cross Validation Error :",cross_val_error(scores),"%")
    
def normalize(data):
    feature_vectors = file_to_vector(data)
    Y = []
    for i in range(len(feature_vectors)):
        Y.append(feature_vectors[i].pop(0))
        for j in range(len(feature_vectors[i])):
            feature_vectors[i][j] = feature_vectors[i][j] * 2 / 255 - 1
    return feature_vectors, Y
    
def file_to_vector(txt):
    f = open(txt,'r')
    vector = []
    for a in f:
        a = a.rstrip('\n').split(',')
        for i in range(len(a)):
            a[i] = int(a[i])
        vector.append(a)
    f.close()
    return vector

def test_error(v1,v2):
    e = 0
    for i in range(len(v1)):
        if v1[i] != v2[i]:
            e += 1
    er = (e / len(v1)) * 100
    return er
    
def cross_val_error(v):
    s = 0
    for n in v:
        s += n
    return 100 - (s/len(v))*100
    
    
    
main()
    
