#data preparation utils

import numpy as np
import tensorflow as tf

def partitionByClass(X,y_true):
    maxc = np.max(y_true+1)
    ids = [[] for i in range(maxc)]
    for i in range(np.shape(y_true)[0]):
        ids[y_true[i]].append(i)
    return ids

def prepareBatch(X,y_true,ids_by_class_train,N_classes = 10, N_support = 10, N_query = 5, permute = True):
    maxc = np.max(y_true) #max class number

    classes = np.random.choice(range(maxc+1), size = (N_classes), replace = False) #choose subset of N_classes classes

    ids_batch = np.array(
        [np.random.choice(ids_by_class_train[c],size = (N_support + N_query), replace = False) for c in classes]
        )

    ids_batch_support = np.ndarray.flatten(ids_batch[:,:N_support])
    ids_batch_query = np.ndarray.flatten(ids_batch[:,N_support:])

    if permute:
        ids_batch_support = np.random.permutation(ids_batch_support)
        ids_batch_query = np.random.permutation(ids_batch_query)

    return X[ids_batch_support,:,:], y_true[ids_batch_support], X[ids_batch_query,:,:], y_true[ids_batch_query], classes

#preprocessing images (loaded background 1.0, character 0.0)
def invert_img(x):
    _,H,W = np.shape(x)
    return -2.0 * np.reshape(x,[-1,H,W]) + 1.0

def deinvert_img(x):
    _,H,W = np.shape(x)
    return 1.0 - 0.5 * x

def resize_img(x,Hold,Wold,Hnew,Wnew):
    q = tf.Session().run(tf.image.resize_images(tf.reshape(x,[-1,Hold,Wold,1]),[Hnew,Wnew]))
    return np.reshape(q,[-1,Hnew,Wnew])

def subtract_mean(X):
    N,H,W = np.shape(X)
    Xf = np.reshape(X,[N,H*W])
    means = np.mean(Xf, axis = 1, keepdims = True)
    Xf = Xf - np.mean(Xf, axis = 1, keepdims = True)
    return np.reshape(Xf,np.shape(X)), means

def augment_by_rotations(X,y,ks = [0,1,2,3]):
    Xs,ys = [],[]
    class_step = np.max(y)+1
    for i,k in enumerate(ks):
        Xs.append(np.rot90(X, k = k, axes = (1,2)))
        ys.append(np.array(y) + (i)*class_step)
    Xa = np.concatenate(Xs,axis = 0)
    ya = np.concatenate(ys,axis = 0)
    return Xa,ya
