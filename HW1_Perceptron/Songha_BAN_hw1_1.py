# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 00:12:31 2017

@author: Songha Ban
"""
import numpy
import pylab

def main():
    N = int(input("Enter N >>"))
    X = int(input("Enter X >>"))
    split_training_data("spam_train.txt", N)
    perceptron_test("train.txt","validation.txt",X)
    #for testing N=5000, use the code below instead of the above four lines
    #split_training_data("spam_train.txt", 5000)
    #perceptron_test("spam_train.txt","spam_test.txt",20)

def split_training_data(data, N):
    emails = file_to_list(data)
    training = emails[0:N]
    validation = emails[N:]
    
    t = open("train.txt", 'w')
    v = open("validation.txt", 'w')     
    for e in training:
        t.write(e + '\n')
    for e in validation:
        v.write(e+'\n')
    t.close()
    v.close()
    
def file_to_list(data):
    f = open(data, 'r')
    contents = f.read()
    f.close()
    return contents.split('\n')
    
def words(data, X):
    emails = file_to_list(data)
    dic = {}
    wordlist = []   
    
    for email in emails:
        temp = set(email[2:].split())
        while len(temp) != 0:
            word = temp.pop()
            if word in dic:
                dic[word] += 1
            else:
                dic[word] = 1
        
    while len(dic) != 0:
        key, value = dic.popitem()
        if value >= X:
            wordlist.append(key)
    print("wordlist :",len(wordlist))
    return wordlist


def feature_vector(email, vocab):
    x = [0] * len(vocab)
    for i in range(len(vocab)):
        if vocab[i] in email:
            x[i] = 1
    return x

def perceptron_train(data, vocab, max_iter):
    print("Perceptron Training")
    X = []
    Y = []
    emails = file_to_list(data)
    for email in emails:
        if len(email) != 0:
            X.append(feature_vector(email, vocab))
            if email[0] == '1':
                Y.append(1)
            else:
                Y.append(-1)
    W = numpy.array([0] * len(vocab))
    X = numpy.array(X)
    Y = numpy.array(Y)
    n_iter = 0
    error = 0
    k = 0
    while True:
        n_iter += 1
        error = 0
        for n in range(len(X)):
            if ((numpy.dot(W, X[n])) * Y[n]) > 0:
                W = W
            else:
                W = numpy.sum([W, numpy.dot(X[n], Y[n])], axis = 0)
                error +=1
        k += error
        print(n_iter, ": error", error)

        if error == 0:
            break
        if n_iter == max_iter:
            break
    return W, k, n_iter
    

def perceptron_error(w, data, vocab):
    X = []
    Y = []
    emails = file_to_list(data)
    for email in emails:
        if len(email) != 0:
            X.append(feature_vector(email, vocab)) 
            if email[0] == '1':
                Y.append(1)
            else:
                Y.append(-1)
    X = numpy.array(X)
    Y = numpy.array(Y)

    error = 0
    for n in range(len(X)):
        if (numpy.dot(w, X[n]) * Y[n]) <= 0:
            error += 1

    return (error/len(X)) * 100

def perceptron_test(Tdata, Vdata, X):
    vocab = words(Tdata, X)
    w, error, iteration = perceptron_train(Tdata, vocab, 2000)
    print("Weight :",w)
    print("Total number of errors :", error)
    print("Number of iterations :", iteration)
    print("Error Rate :", perceptron_error(w, Vdata, vocab),"%")
    negative, positive = twelve_words(w, vocab)
    print("12 words with the most positive weights :", positive)
    print("12 words with the most negative weights :", negative)
    
def twelve_words(w, vocab):
    words = []
    positive = []
    negative = []
    n = len(vocab)
    
    for i in range(n):
        words.append((w[i], vocab[i]))
    
    words.sort()
    for pair in words[:12]:
        negative.append(pair[1])
    for pair in words[n-12:]:
        positive.append(pair[1])

    return negative, positive
    
def plot_iter():
    pylab.xlim(0,5000)
    pylab.ylim(0,25)
    
    pylab.xlabel('N (rows of training data)')
    pylab.ylabel('Number of Iterations')
    pylab.plot([100,200,400,800,2000,4000],[17,7,15,8,11,13], label = 'iteration')
    
    pylab.legend()
    pylab.show()
    
def plot_error():
    pylab.xlim(0,5000)
    pylab.ylim(0,25)
    
    pylab.xlabel('N (rows of training data)')
    pylab.ylabel('Error Rate')
    pylab.plot([100,200,400,800,2000,4000],[20.16,10.29,5.83,4.16,2.87,1.80], label = 'error')
    
    pylab.legend()
    pylab.show()
      
    
main()
plot_iter()
plot_error()