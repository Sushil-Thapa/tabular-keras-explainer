"""
This script contains functions for generating synthetic data. 

Part of the code is based on: 
Learning to Explain paper: https://arxiv.org/pdf/1802.07814.pdf

https://github.com/Jianbo-Lab/CCM, 
https://github.com/Jianbo-Lab/L2X/blob/master/synthetic/make_data.py
""" 
import numpy as np  
from scipy.stats import chi2
import tensorflow as tf

from settings import SEED_VALUE


np.random.seed(SEED_VALUE)

def generate_random_labels(X, c = 10):
    # This generates random labels, Primarily for debug/dummy purpose.
    # tries to make labels max(n,c) where n is number of instances in X (modulo operator)
    y = np.random.randint(X.shape[0]%c, size = X.shape[0])
    return y

def generate_sum_labels(X):
    # This generates label 1 when X has positive sum for all feats, 0 otherwise
    y = np.where(np.sum(X, axis=1) > 0, 1, 0)
    y = tf.keras.utils.to_categorical(y)
    return y

    # just exploring random label setup
    # thres = np.where(X > 0, 1, 0)
    # uniq, cnts= np.unique(thres, return_counts=1, axis=1)
    # y = uniq[np.argmax(cnts)]


def generate_XOR_labels(X):
    y = np.exp(X[:,0]*X[:,1])

    prob_1 = np.expand_dims(1 / (1+y) ,1)
    prob_0 = np.expand_dims(y / (1+y) ,1)

    y = np.concatenate((prob_0,prob_1), axis = 1)

    return y

def generate_orange_labels(X):
    logit = np.exp(np.sum(X[:,:4]**2, axis = 1) - 4.0) 
    prob_1 = np.expand_dims(1 / (1+logit) ,1)
    prob_0 = np.expand_dims(logit / (1+logit) ,1)

    y = np.concatenate((prob_0,prob_1), axis = 1)

    return y

def generate_additive_labels(X):
    logit = np.exp(-100 * np.sin(0.2*X[:,0]) + abs(X[:,1]) + X[:,2] + np.exp(-X[:,3])  - 2.4) 

    prob_1 = np.expand_dims(1 / (1+logit) ,1)
    prob_0 = np.expand_dims(logit / (1+logit) ,1)

    y = np.concatenate((prob_0,prob_1), axis = 1)

    return y



def generate_data(args, val = False):
    """
    Generate data (X,y)
    Args:
        n(int): number of samples 
        datatype(string): The type of data 
        choices: 'random', 'positive', 'orange_skin', 'XOR', 'regression'.
        seed: random seed used
    Return: 
        X(float): [n,d].  
        y(float): n dimensional array. 
    """
    n = args.n_samples
    f = args.n_feats
    datatype = args.datatype

    X = np.random.randn(n, f)

    datatypes = None 
    if datatype == 'random': 
        y = generate_random_labels(X) 

    elif datatype == 'sum': 
        y = generate_sum_labels(X) 

    elif datatype == 'orange_skin': 
        y = generate_orange_labels(X) 

    elif datatype == 'XOR':
        y = generate_XOR_labels(X)    

    elif datatype == 'nonlinear_additive':  
        y = generate_additive_labels(X) 

    elif datatype == 'switch':

        # Construct X as a mixture of two Gaussians.
        X[:n//2,-1] += 3
        X[n//2:,-1] += -3
        X1 = X[:n//2]; X2 = X[n//2:]

        y1 = generate_orange_labels(X1)
        y2 = generate_additive_labels(X2)

        # Set the key features of X2 to be the 4-8th features.
        X2[:,4:8],X2[:,:4] = X2[:,:4],X2[:,4:8]

        X = np.concatenate([X1,X2], axis = 0)
        y = np.concatenate([y1,y2], axis = 0) 

        # Used for evaluation purposes.
        datatypes = np.array(['orange_skin'] * len(y1) + ['nonlinear_additive'] * len(y2)) 

        # Permute the instances randomly.
        perm_inds = np.random.permutation(n)
        X,y = X[perm_inds],y[perm_inds]
        datatypes = datatypes[perm_inds]


    return X, y, datatypes  
