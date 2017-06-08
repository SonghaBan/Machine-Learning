# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 22:29:26 2017

@author: song-isong-i
"""

import csv
import math

def main():
    x = file_to_list("ps5_data.csv")
    y = label(file_to_list("ps5_data-labels.csv"))
    w1 = file_to_list("ps5_theta1.csv")
    w2 = file_to_list("ps5_theta2.csv")
    print("forward error rate :",forward_propagation(x,y,w1,w2))
    print(MLEcostf(x,y,w1,w2,10))
    
        
def label(y):
    for i in range(len(y)):
        y[i] = int(y[i][0]) - 1
    return y    

def gx(x):
    if x<0:
        return 1 - 1 / (1 + math.exp(x))
    return 1/(1+math.exp(-x))
    
def dotp(v1, v2):
    s = 0
    for i in range(len(v1)):
        s += v1[i] * v2[i]
    return s                
        
def file_to_list(file):
    csvlist = []   
    f = open(file, 'r')
    contents = csv.reader(f)
    for c in contents:
        for i in range(len(c)):
            c[i] = float(c[i])
        csvlist.append(c)
    f.close()
    return csvlist
    
def neuron_unit(a,wi):
    r = wi[0]
    r += dotp(a,wi[1:])
    return gx(r)
    
def neural_network(xi, w1, w2):
    a = []
    output = []
    for wi in w1:
        a.append(neuron_unit(xi,wi))
    for wi in w2:
        output.append(neuron_unit(a,wi))
    return output
    
def classify(xi, w1, w2):
    output = neural_network(xi, w1, w2)
    return maximum(output)
    
def maximum(l):
    m = 0
    for i in range(1,len(l)):
        if l[m] < l[i]:
            m = i
    return m

def error(result, label):
    e = 0
    for i in range(len(result)):
        if result[i] != label[i]:
            e += 1
    return e*100/len(result)

def forward_propagation(x,y,w1,w2):
    result = []
    for l in x:
        result.append(classify(l,w1,w2))
    return error(result, y)
    
def label_to_vector(yi,nk):
    y = [0] * nk
    y[yi] = 1
    return y

def MLEcostf(x,y,w1,w2,nk):  
    s = 0
    m = len(x)
    for i in range(m):
        temp = 0
        hxi = neural_network(x[i],w1,w2)
        yi = label_to_vector(y[i],nk)
        for k in range(nk):
            temp += yi[k] * math.log(hxi[k]) + (1-yi[k]) * math.log(1-hxi[k])
        s += temp
    return -s/m

def transpose(matrix):
    new = []
    for i in range(len(matrix[0])):
        new.append([0] * len(matrix))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            new[j][i] = matrix[i][j]
    return new
    

main()
