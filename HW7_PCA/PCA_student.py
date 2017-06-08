import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random

######### Read the data ##########

infile = open('faces.csv','r')
img_data = infile.read().strip().split('\n')
faces = np.reshape([list(map(int,a.strip().split(','))) for a in img_data],(400,4096))

######### Global Variable ##########

image_count = 0

######### Function that normalizes a vector x (i.e. |x|=1 ) #########

# > numpy.linalg.norm(x, ord=None, axis=None, keepdims=False) 
#   This function is able to return one of eight different matrix norms, 
#   or one of an infinite number of vector norms (described below), 
#   depending on the value of the ord parameter.

def normalize(U):
	return U / LA.norm(U) 

######### Display first face #########

# Useful functions:
# > numpy.reshape(a, newshape, order='C')
#   Gives a new shape to an array without changing its data.
# > matplotlib.pyplot.figure()
# 	Creates a new figure.
# > matplotlib.pyplot.title()
#	Set a title of the current axes.
# > matplotlib.pyplot.imshow()
#	Display an image on the axes.
#	Note: You need a matplotlib.pyplot.show() at the end to display all the figures.

def display_firstface():
    first_face = np.reshape(faces[0],(64,64),order='F')
    global image_count 
    image_count += 1
    plt.figure(image_count)
    plt.title('First_face')
    plt.imshow(first_face,cmap=plt.cm.gray)


########## display a random face ###########

# Useful functions:
# > numpy.random.choice(a, size=None, replace=True, p=None)
#   Generates a random sample from a given 1-D array
# > numpy.ndarray.shape()
#   Tuple of array dimensions.

#### Your Code Here ####
def display_randomface():
    random_face = np.reshape(faces[random.randint(0,399)],(64,64),order = 'F')
    global image_count 
    image_count += 1
    plt.figure(image_count)
    plt.title('Random_face')
    plt.imshow(random_face,cmap=plt.cm.gray)


########## compute and display the mean face ###########

# Useful functions:
# > numpy.mean(a, axis='None', ...)
#   Compute the arithmetic mean along the specified axis.
#   Returns the average of the array elements. The average is taken over 
#   the flattened array by default, otherwise over the specified axis. 
#   float64 intermediate and return values are used for integer inputs.

#### Your Code Here ####
def meanface():
    meanface = []
    for i in range(len(faces[0])):
        meanface.append(np.mean([f[i] for f in faces]))    
    return meanface

def display_meanface(meanface):
    mean_face = np.reshape(meanface,(64,64),order = 'F')
    global image_count 
    image_count += 1
    plt.figure(image_count)
    plt.title('Mean_face')
    plt.imshow(mean_face,cmap=plt.cm.gray)

    
######### substract the mean from the face images and get the centralized data matrix A ###########

# Useful functions:
# > numpy.repeat(a, repeats, axis=None)
#   Repeat elements of an array.

#### Your Code Here ####
def get_centralized(m):
    return np.array([f for f in faces - m])
    
#question 3
def average_covM(A):
    avgV = np.matrix(np.reshape([0]*4096,(64,64),order = 'F'))
    for cf in A:
        x = np.matrix(np.reshape(cf,(64,64),order = 'F'))
        avgV = avgV + np.dot(x, x.transpose())
    return avgV / 400

######### calculate the eigenvalues and eigenvectors of the covariance matrix #####################

# Useful functions:
# > numpy.matrix()
#   Returns a matrix from an array-like object, or from a string of data. 
#   A matrix is a specialized 2-D array that retains its 2-D nature through operations. 
#   It has certain special operators, such as * (matrix multiplication) and ** (matrix power).

# > numpy.matrix.transpose(*axes)
#   Returns a view of the array with axes transposed.

# > numpy.linalg.eig(a)[source]
#   Compute the eigenvalues and right eigenvectors of a square array.
#   The eigenvalues, each repeated according to its multiplicity. 
#   The eigenvalues are not necessarily ordered. 

#### Your Code Here ####                                   ++++++++++++++++

def PCA(A,nPC):
    L = np.dot(A,A.transpose())   #(400 * 400)
    evals, evecs = np.linalg.eig(L)
    V = np.dot(A.transpose(),evecs)    
    for i in range(400):
        V[:,i] = V[:,i]/np.linalg.norm(V[:,i])
    idx = np.argsort(-evals)
    evals = evals[idx]
    V = V[:,idx]
    return evals[:nPC],V[:,:nPC]

    

########## Display the first 10 principal components ##################

#### Your Code Here ####
def first_10PC(A):
    global image_count 
    image_count += 1
    fig, ax = plt.subplots(nrows=3, ncols=3, num = image_count)
    vals10, PC10 = PCA(A,10)
    PC10 = PC10.transpose()
    i = 0
    for row in ax:       
        for col in row:
            face = np.reshape(PC10[i], (64,64), order = 'F')
            col.imshow(face,cmap=plt.cm.gray)
            i += 1



########## Reconstruct the first face using the first two PCs #########

#### Your Code Here ####

def reconstruct(A, m, nPC,face,display,title = ''): #display is boolean
    vals, U = PCA(A,nPC)
    eigenface = np.dot(U.transpose(),(face - m))
    projected = m + np.dot(U,eigenface)
    if display:
        pf = np.reshape(projected,(64,64),order = 'F')
        global image_count
        image_count+=1
        plt.figure(image_count)
        plt.title(title)
        plt.imshow(pf,cmap=plt.cm.gray)
    return projected

def face1_2PC(A, m):
    reconstruct(A,m,2,faces[0],True,'First_face with 2 PCs')

########## Reconstruct random face using the first 5, 10, 25, 50, 100, 200, 300, 399  PCs ###########

#### Your Code Here ####
def test_reconstruct(A,m):
    test = [5, 10, 25, 50, 100, 200, 300, 399]
    for n in test:
        face = faces[random.randint(0,399)]
        reconstruct(A,m,n, face, True, 'Random_face with '+str(n)+' PCs')

#test_reconstruct()


######### Plot proportion of variance of all the PCs ###############

# Useful functions:
# > matplotlib.pyplot.plot(*args, **kwargs)
#   Plot lines and/or markers to the Axes. 
# > matplotlib.pyplot.show(*args, **kw)
#   Display a figure. 
#   When running in ipython with its pylab mode, 
#   display all figures and return to the ipython prompt.

#### Your Code Here ####
def plot_variance(A):
    evals, evectors = PCA(A,400)
    p_variance = evals / np.sum(evals)
    global image_count
    image_count += 1
    plt.figure(image_count)
    plt.ylim(0,0.25)
    #plt.acorr(p_variance)
    x = range(400)
    plt.plot(x,p_variance)

def main():
    display_firstface()
    display_randomface()
    m = meanface()
    display_meanface(m)
    A = get_centralized(m)
    first_10PC(A)
    face1_2PC(A,m)
    test_reconstruct(A,m)
    plot_variance(A)
    
main()


