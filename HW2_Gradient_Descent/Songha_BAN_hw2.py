# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 18:03:05 2017

@author: song-isong-i
"""
import numpy
import pylab
import random

def main():
    normalize("housing.txt")
    j1, w1 = gradient_descent("normalized.txt", 0.05, 80)
    j2, w2 = gradient_descent("normalized.txt", 0.5, 80)
    j3, w3 = gradient_descent("normalized.txt", 1.5, 80)
    print(w1,w2,w3)
    print(j1,j2,j3)
    
    print("Predicted Price :",predict(1650,3,w1))  
    
    plot(0.01, 'alpha = 0.01')
    plot(0.1, 'alpha = 0.1')
    plot(0.3, 'alpha = 0.3')
    
    Jv, Wv = gradient_descent("normalized.txt", 0.1, 80)    
    Js, Ws = stochastic_gradient("normalized.txt",0.1)
    print("Compare Vanilla and Stochastic")
    print("Vanilla : (w0, w1, w2) =", Wv)
    print("Stochastic : (w0, w1, w2) =", Ws)

def mean_sd(L):
    s = 0
    sd = 0
    for n in L:
        s += n
    mean = s/len(L)
    for n in L:
        sd += (n - mean)**2
    sd = (sd/len(L))**0.5
    return mean, sd
    
def normalize(txt):
    housing_list = file_to_list(txt)
    x1 = []
    x2 = []
    y = []
    for d in housing_list:
        if len(d) != 0:
            a = d.split(',')
            x1.append(int(a[0]))
            x2.append(int(a[1]))
            y.append(int(a[2]))
    mean1, sd1 = mean_sd(x1)
    mean2, sd2 = mean_sd(x2)
    meany, sdy = mean_sd(y) 
    
    g = open("normalized.txt", 'w')
    for n in range(len(x1)):
        x11 = (x1[n] - mean1) / sd1
        x22 = (x2[n] - mean2) / sd2
        ny = (y[n] - meany) / sdy
        g.write(str(x11)+','+str(x22)+','+str(ny)+'\n')
    g.close()
    return mean1, sd1, mean2, sd2, meany, sdy

def Jw(x, y, w):
    J = 0
    m = len(x)
    for i in range(m):
        J += (numpy.dot(w, x[i]) - y[i]) ** 2 / (2*m)
    return J
    
def gradient_descent(data, alpha, passes):
    w = [0,0,0]
    datalist = file_to_list(data)
    x = []
    y = []
        
    for n in range(len(datalist)):
        if len(datalist[n]) != 0:
            a = datalist[n].split(',')
            x.append([1, float(a[0]), float(a[1])])
            y.append(float(a[2]))
    
    w2 = w
    for n in range(passes):
        for j in range(len(w)): # calculate sigma. w should not change
            gradient = 0
            for i in range(len(x)): # save updated values of w in w2. When finished updating, replace w with w2.
                loss = numpy.dot(w,x[i]) - y[i]
                gradient += numpy.dot(x[i][j], loss) / len(y)
            w2[j] = w2[j] - alpha * gradient
        w = w2
        J = Jw(x, y, w)
    return J,w
        
    
def stochastic_gradient(data, alpha):
    w = [0] * 3
    datalist = file_to_list(data)
    x = []
    y = []
    w2 = w
    
    for n in range(len(datalist)):
        if len(datalist[n]) != 0:
            a = datalist[n].split(',')
            x.append(numpy.array([1, float(a[0]), float(a[1])]))
            y.append(float(a[2]))     
    
    for n in range(3):
        for i in range(len(x)): 
            loss = numpy.dot(w,x[i]) - y[i]
            for j in range(len(w)): 
                gradient = x[i][j] * loss / len(y)
                w2[j] = w2[j] - alpha * gradient
            w = w2
        J = Jw(x, y, w)
        print(n+1,":", J)
        x,y = shuffle(x,y)
        
    return J,w
    
def file_to_list(data):
    f = open(data, 'r')
    contents = f.read()
    f.close()
    return contents.split('\n')
    
def predict(area, bedrooms, w):
    mean1, sd1, mean2, sd2, meany, sdy = normalize("housing.txt")
    area = (area - mean1) / sd1
    bedrooms = (bedrooms - mean2) / sd2
    x = numpy.array([1, area, bedrooms])
    price = numpy.dot(w, x)
    price = price * sdy + meany
    return price
    
def shuffle(x,y):
    new = []
    for n in range(len(y)):
        a = x[n].tolist()
        new.append(a)
    for n in range(len(y)):
        new[n].append(y[n])
    random.shuffle(new)
    for n in range(len(y)):
        y[n] = new[n].pop()
        
    new = numpy.array(new)
    return new,y
    
    
def plot(alpha, l):
    pylab.xlabel('iterations')
    pylab.ylabel('J(w)')
    iterations = [10,20,30,40,50,60,70,80]
    J = []
    for n in iterations:
        j, w = gradient_descent("normalized.txt", alpha, n)
        J.append(j)
    print(J)
    pylab.plot(iterations, J, label = l)
    pylab.legend()
    pylab.show()

    

main()
    


