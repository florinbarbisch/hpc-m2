import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

NP_SHARED_NAME = 'npshared'
NP_DATA_TYPE = np.float64
NP_SIZE = (u.shape[0], vt.shape[1])

reco_shared_mem = SharedMemory(create=True, size=int(np.prod(NP_SIZE) * np.dtype(NP_DATA_TYPE).itemsize), name=NP_SHARED_NAME)
print(f"shared memory created: {reco_shared_mem.name}")
reco = np.ndarray(NP_SIZE, dtype=NP_DATA_TYPE, buffer=reco_shared_mem.buf)
print(f"shared memory ndarray created: {reco.shape}")

def reconstruct_svd_mp(u,s,vt,k,i):
    """SVD reconstruction for k components for the i-th row using multiprocessing
    
    Inputs:
    u: (m,n) numpy array
    s: (n) numpy array (diagonal matrix)
    vt: (n,n) numpy array
    k: number of reconstructed singular components
    i: row index
    """
    reco_shared_mem = SharedMemory(NP_SHARED_NAME)
    reco = np.ndarray(NP_SIZE, dtype=NP_DATA_TYPE, buffer=reco_shared_mem.buf)
    print(f"reconstructing row {i}", flush=True)
    reco[i] = ((u[i,:k] * s[:k]) @ vt[:k,:])


def task(u,s,vt,k,i):
    print(f"reconstructing row {i}", flush=True)


print("start multiprocessing")

with mp.Pool(processes=os.cpu_count()) as pool:
    print(f"pool created with {os.cpu_count()} processes")

    # submit all jobs
    _ = pool.map_async(task, [(u,s,vt,30, i) for i in range(u.shape[0])])
    print("all jobs submitted")

    # close the pool
    pool.close()
    print("pool closed")
    # wait for all tasks to complete
    pool.join()
    print("pool joined")


    # display results
    #plot_reco(reco._reco, 30)

reco_shared_mem.close()
reco_shared_mem.unlink()
### END SOLUTION