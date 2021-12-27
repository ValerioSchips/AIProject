import numpy as np
import pandas as pd
import sklearn
import subprocess
import gzip
import json
import os
from collections import Counter
from datetime import datetime
import math
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse import lil_matrix
import copy
import bottleneck as bn
from sklearn.metrics import ndcg_score
from sklearn.metrics import recall_score


def Recall_ndcg_at_k(X_pred, test_mat, k):

    #recall
    idx = np.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    for x in range(X_pred_binary.shape[0]):
        X_pred_binary[x, idx[x, :k]] = True

    X_true_binary = (test_mat > 0)
    tmp = (np.logical_and(X_true_binary, X_pred_binary)).astype(np.int32)

    den = np.minimum(k, X_true_binary.sum(axis=1)).astype(np.float32)
    print(len(den))

    if not np.all(tmp.sum(axis=1)<=k) and not np.all(tmp.sum(axis=1)>=0):
        print("error in k = ", k)

    recall = tmp.sum(axis=1) / den
    recall = np.nan_to_num(recall, nan=0.0)

    # ndcg
    my_array = np.zeros((tmp.shape[0], k))
    for x in range(tmp.shape[0]):
        my_array[x, :] = (tmp[x, idx[x, :k]])
    
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (my_array * tp).sum(axis=1)
    IDCG = np.zeros_like(my_array)
    indexes = np.minimum(k, my_array.sum(axis=1)).astype(np.int32)
    for x in range(DCG.shape[0]):
        IDCG[x, :indexes[x]] = 1.0

    IDCG = (IDCG * tp).sum(axis=1)
    out = DCG / IDCG
    out = np.nan_to_num(out, nan=0.0)

    return recall, out

def evaluate(mytest="", target="", path = ""):

    BB = np.load(path + f"BB{target}.npy")
    CC = np.load(path + f"CC{target}.npy")
    X_test = sparse.load_npz(path + f"X_test{mytest}.npz")
    Z_test = sparse.load_npz(path + f"Z_test{mytest}.npz")

    print(BB.shape)
    print(CC.shape)
    print(X_test.shape)
    print(Z_test.shape)
    print(X_test.nnz)
    print(Z_test.nnz)
    test_te = pd.read_csv(path + f"test_te{mytest}.csv")
    test_te = test_te[['user_id', 'item_id']]
    #pred_val = (X_test).dot(BB) + Z_test.dot(CC)
    pred_val = (X_test @ BB) + (Z_test @ CC)
    pred_val[X_test.nonzero()] = -np.inf # exclude examples from training and validation (if any)
    print(pred_val.shape)
    print(pred_val.nonzero()[0].shape)
    test_matrix = np.zeros_like(pred_val, dtype=bool)

    for _, row in test_te.iterrows():
        test_matrix[row['user_id'], row['item_id']] = 1

    print(test_matrix.nonzero()[0].shape)

    #r3 = (Recall_at_k(pred_val, test_matrix, k=3))
    r5, NDCG5 = (Recall_ndcg_at_k(pred_val, test_matrix, k=5))
    r10, NDCG10 = (Recall_ndcg_at_k(pred_val, test_matrix, k=10))
    #r20 = (Recall_at_k(pred_val, test_matrix, k=20))
    #r40 = (Recall_at_k(pred_val, test_matrix, k=40))

    #print(np.mean(r3), "std error: ", np.std(r3) / np.sqrt(len(r3)))
    print("Recall")
    print(np.mean(r5), "std error: ", np.std(r5) / np.sqrt(len(r5)))
    print(np.mean(r10), "std error: ", np.std(r10) / np.sqrt(len(r10)))
    #print(np.mean(r20), "std error: ", np.std(r20) / np.sqrt(len(r20)))
    #print(np.mean(r40),  "std error: ", np.std(r40) / np.sqrt(len(r40)))
    # 7. Interesting Plots
    #from matplotlib import pyplot as plt 

    #a = pred_val[np.where(pred_val > -3.0)].flatten()
    #plt.figure(dpi=300)
    #plt.hist(a, bins=500, log=True)
    #plt.title("Predicted Values Distribution") 
    #plt.axvline(x=0.0, label='0', c='r', linewidth=0.1)
    #plt.savefig(path + "fig.png")

    #pred_val[X_test.nonzero()] = -10000

    print("NDCG ", np.mean(NDCG5), np.mean(NDCG10))
    return np.mean(r5), np.mean(r10), np.mean(NDCG5), np.mean(NDCG10)


if __name__ == "__main__":
    evaluate(target="")