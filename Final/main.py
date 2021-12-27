
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # AI Recomender Project

# %%
import os
os.environ['MKL_NUM_THREADS'] = '64'
os.environ['OMP_NUM_THREADS'] = '64' 
os.environ['CUPY_TF32'] = '1' 

import numpy as np
import pandas as pd
from scipy.sparse import csr
import sklearn, sys
import subprocess
import gzip
import json
from collections import Counter
from datetime import datetime
import math
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse import lil_matrix
import copy
import train_algo
import sparse_dot_mkl
import evaluate

# computing M
def compute_M(X, pair_list):
    M = lil_matrix((len(pair_list), X.shape[1]), dtype=np.float32)

    for x in range(len(pair_list)):
        M[x, pair_list[x, 0]] = 1
        M[x, pair_list[x, 1]] = 1

    M = csr_matrix(M)
    return M

def start(rho, lambdaBB, lambdaCC, t=45, target="", out=""):

    test_te = pd.read_csv(f"test_te{target}.csv")
    test_tr = pd.read_csv(f"test_tr{target}.csv")
    train = pd.read_csv(f"train{target}.csv")

    try :
        sparse.load_npz(f"X{target}.npz")
        print("X found")
    except:
        max_item_id = max(test_te["item_id"].max(), test_tr["item_id"].max(), train["item_id"].max())
        print(f"max item id = {max_item_id}")

        X = np.zeros((train["user_id"].max()+1, max_item_id+1), dtype=np.bool_)

        for _, row in train.iterrows():
            X[row['user_id'], row['item_id']] = 1

        X = csr_matrix(X, dtype=np.float32)

        sparse.save_npz(f"X{target}.npz", X)

        X_test = np.zeros((test_tr["user_id"].max()+1, max_item_id+1), dtype=np.bool_)

        for _, row in test_tr.iterrows():
            X_test[row['user_id'], row['item_id']] = 1

        X_test = csr_matrix(X_test, dtype=np.float32)

        sparse.save_npz(f"X_test{target}.npz", X_test)


    X = sparse.load_npz(f"X{target}.npz")
    X_test = sparse.load_npz(f"X_test{target}.npz")

    X_transpose = csr_matrix(X.T, dtype=np.float32)
    X = csr_matrix(X, dtype=np.float32)
    X_test = csr_matrix(X_test, dtype=np.float32)

    # identifyig valuable pairs
    print("Using mkl gram")
    XtX = sparse_dot_mkl.gram_matrix_mkl(X, transpose=False) #X_transpose @ X
    print("Computed mkl gram")
    XtXdiag = XtX.diagonal()
    XtX = XtX.todense()
    XtX[np.diag_indices(XtX.shape[0])] = 0.0
    XtX = csr_matrix(XtX)
    print("Check: ", np.all(XtX[np.tril_indices(XtX.shape[0], k=0)] == 0.0))

    pair_list = np.argwhere(XtX > t)

    print("using mkl dot")
    XtX = sparse_dot_mkl.dot_product_mkl(X_transpose, X)

    pair_list = np.array(pair_list, dtype=np.int32)
    print(pair_list.shape)

    M = compute_M(X, pair_list)
    # computing Z
    print("computing Z")
    Z = sparse_dot_mkl.dot_product_mkl(X, M.transpose()) #X @ M.transpose()
    print(f"Z computed: number of entries {Z.nnz}, shape: {Z.shape}")

    # non linear function f
    Z = (Z == 2.0 )

    # back to float 64 from np.bool_
    Z = csr_matrix(Z, dtype=np.float32)

    # computing Z_test 
    Z_test = sparse_dot_mkl.dot_product_mkl(X_test, compute_M(X_test, pair_list).transpose()) #X_test @ compute_M(X_test, pair_list).transpose()
    Z_test = (Z_test == 2.0 )
    Z_test = csr_matrix(Z_test, dtype=np.float32)

    sparse.save_npz(f"Z{target}.npz", Z)
    sparse.save_npz(f"Z_test{target}.npz", Z_test)
    Zt = Z.T
    Zt.indptr = Zt.indptr.astype(np.uint64)
    Zt.indices = Zt.indices.astype(np.uint64)
    ZtZ = sparse_dot_mkl.dot_product_mkl(Zt, Z)# Zt.dot(Z)
    ZtX = sparse_dot_mkl.dot_product_mkl(Zt, X)# Zt.dot(X)
    ZtZdiag=copy.deepcopy(ZtZ.diagonal())

    max_epochs = 30

    print("Pre-processing done")

    BB, CC, loss= train_algo.train(M, X, Z, XtX, ZtZ, ZtX, ZtZdiag, XtXdiag, max_epochs, rho, lambdaBB, lambdaCC)

    np.save(f"./CC{out}.npy", CC)
    np.save(f"./BB{out}.npy", BB)
    np.save(f"./loss{out}.npy", np.array(loss))


if __name__ == '__main__':

    start(rho, lambdaBB, lambdaCC, t=thresh, target=f"_old_data{my_iterator}", out=f"_old_data{my_iterator}_{rho}_{lambdaCC}_{lambdaBB}")
    r5, r10, NDCG5, NDCG10 = evaluate.evaluate(mytest=f"_old_data{my_iterator}", target=f"_old_data{my_iterator}_{rho}_{lambdaCC}_{lambdaBB}")
    