import numpy as np
import pandas as pd
from scipy import stats
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
from scipy.linalg import norm
import copy
import sparse_dot_mkl
import cupy as cp
import cupyx as cpx

def train(M, X, Z, XtX, ZtZ, ZtX, ZtZdiag, XtXdiag, max_epochs, rho, lambdaBB, lambdaCC):
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    print("precomputing BB")
    # precompute for BB
    ii_diag = np.diag_indices(XtX.shape[0], ndim=2)
    XtX = XtX.todense()
    XtX[ii_diag] = XtXdiag+lambdaBB

    print("using the gpu")
    XtX_gpu = cp.asarray(XtX)
    PP_gpu = cp.linalg.inv(XtX_gpu)

    PP = cp.asnumpy(PP_gpu)
    XtX[ii_diag] = XtXdiag
    del XtX_gpu, PP_gpu
    cp._default_memory_pool.free_all_blocks()
    print("gpu task done")

    XtX = sparse.csr_matrix(XtX, dtype=np.float32)

    print("precomputing CC")
    # precompute for CC
    ii_diag_ZZ=np.diag_indices(ZtZ.shape[0], ndim=2)
    ZtZ = ZtZ.todense()
    ZtZ[ii_diag_ZZ] = ZtZdiag+lambdaCC+rho

    print("using the gpu")
    ZtZ_gpu = cp.asarray(ZtZ)
    QQ_gpu = cp.linalg.inv(ZtZ_gpu)
    
    QQ = cp.asnumpy(QQ_gpu)
    del ZtZ_gpu, QQ_gpu
    cp._default_memory_pool.free_all_blocks()
    print("gpu task done")

    ZtZ = sparse.csr_matrix(ZtZ, dtype=np.float32)

    # initialize
    loss = np.zeros(max_epochs)
    CC = np.zeros( (ZtZ.shape[0], XtX.shape[0]),dtype=np.float32 )
    BB = np.zeros( (XtX.shape[0], XtX.shape[0]),dtype=np.float32 )
    DD = np.zeros( (ZtZ.shape[0], XtX.shape[0]),dtype=np.float32 )
    UU = np.zeros( (ZtZ.shape[0], XtX.shape[0]),dtype=np.float32 ) # is Gamma in paper
    element_count = XtX.shape[0] * ZtZ.shape[0]
    #ZtX = ZtX.todense()
    CCmask_cpu = 1.0 - M.todense()


    # Moving on the gpu 
    print("moving data to gpu")
    XtX_gpu = cpx.scipy.sparse.csr_matrix(XtX, dtype=cp.float32)
    ZtX_gpu = cpx.scipy.sparse.csr_matrix(ZtX, dtype=cp.float32)
    X_gpu = cpx.scipy.sparse.csr_matrix(X, dtype=cp.float32)
    Z_gpu = cpx.scipy.sparse.csr_matrix(Z, dtype=cp.float32)
    CC_gpu = cp.asarray(CC, dtype=cp.float32)
    BB_gpu = cp.asarray(BB, dtype=cp.float32)
    loss_gpu = cp.asarray(loss, dtype=cp.float32)

    for iter in range(max_epochs):
        print("\nepoch {}".format(iter))

        DD_gpu = cp.asarray(DD, dtype=cp.float32)
        UU_gpu = cp.asarray(UU, dtype=cp.float32)

        # learn BB
        print("Learn BB with gpu")
        PP_gpu = cp.asarray(PP, dtype=cp.float32)

        BB_gpu[:] = PP_gpu.dot(XtX_gpu-ZtX_gpu.T.dot(CC_gpu))
        #gamma = cp.diag(BB_gpu) / cp.diag(PP_gpu)
        BB_gpu[:] -= cp.multiply(PP_gpu, (cp.diag(BB_gpu) / cp.diag(PP_gpu)))

        #BB = cp.asnumpy(BB_gpu)
        del PP_gpu
        #cp._default_memory_pool.free_all_blocks()

        # learn CC
        print("Learn CC with gpu")
        QQ_gpu = cp.asarray(QQ, dtype=cp.float32)

        CC_gpu[:] = QQ_gpu.dot(ZtX_gpu-ZtX_gpu.dot(BB_gpu) +rho *(DD_gpu-UU_gpu))

        #CC = cp.asarray(CC_gpu)
        del QQ_gpu,

        # learn DD
        print("Learn DD")
        #temp = CC_gpu.copy()
        #temp[CCmask] = 0.0
        CCmask = cp.asarray(CCmask_cpu, dtype=cp.float32)
        DD_gpu[:] = cp.multiply(CC_gpu, CCmask)
        #print("After CCmask ", cp.count_nonzero(DD_gpu))
        #DD= np.maximum(0.0, DD) # if you want to enforce non-negative parameters
        # learn UU (is Gamma in paper)

        UU_gpu[:] += CC_gpu-DD_gpu

        DD = cp.asnumpy(DD_gpu)
        UU = cp.asnumpy(UU_gpu)

        # computing nonzero elements count:
        print(f"non zeros\n CC {cp.count_nonzero(CC_gpu)/element_count}\n BB {cp.count_nonzero(BB_gpu)/XtX.shape[0]**2}")
        print(f" DD {cp.count_nonzero(DD_gpu)/element_count}")
        print(f" UU {cp.count_nonzero(UU_gpu)/element_count}")

        del DD_gpu, UU_gpu, CCmask
        
        print("Memory info")
        print(mempool.used_bytes())              # 0
        print(mempool.total_bytes())             # 0
        print(pinned_mempool.n_free_blocks())    # 0

        cp._default_memory_pool.free_all_blocks()


        print("Freed Memory info")
        print(mempool.used_bytes())              # 0
        print(mempool.total_bytes())             # 0
        print(pinned_mempool.n_free_blocks())    # 0
        # computing loss
        if True:

            loss_gpu[iter] = cp.linalg.norm(X_gpu - X_gpu.dot(BB_gpu) - Z_gpu.dot(CC_gpu))**2 + lambdaBB*cp.linalg.norm(BB_gpu)**2 + lambdaCC*cp.linalg.norm(CC_gpu)**2
            print(f"loss = {loss_gpu[iter]}")


            if iter > 3:
                y = cp.asnumpy(loss_gpu[iter-4:iter])
                x = [0, 1, 2, 3]
                res = stats.linregress(x, y)
                print(f"slope = {res.slope}")
                if res.slope >= -0.16:
                    #pass
                    return cp.asnumpy(BB_gpu), cp.asnumpy(CC_gpu), cp.asnumpy(loss_gpu)


    return cp.asnumpy(BB_gpu), cp.asnumpy(CC_gpu), cp.asnumpy(loss_gpu)