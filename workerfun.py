import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory


def reconstruct_svd_mp(u,s,vt,k,i):
    """SVD reconstruction for k components for the i-th row using multiprocessing
    
    Inputs:
    u: (m,n) numpy array
    s: (n) numpy array (diagonal matrix)
    vt: (n,n) numpy array
    k: number of reconstructed singular components
    i: row index
    """
    
    NP_SHARED_NAME = 'npshared21111'
    NP_DATA_TYPE = np.float64
    NP_SIZE = (u.shape[0], vt.shape[1])
    reco_shared_mem = SharedMemory(size=int(np.prod(NP_SIZE) * np.dtype(NP_DATA_TYPE).itemsize), name=NP_SHARED_NAME)
    reco = np.ndarray(NP_SIZE, dtype=NP_DATA_TYPE, buffer=reco_shared_mem.buf)
    print(f"reconstructing row {i}", flush=True)
    reco[i] = ((u[i,:k] * s[:k]) @ vt[:k,:])


def task(u,s,vt,k,i):
    print(f"reconstructing row {i}", flush=True)