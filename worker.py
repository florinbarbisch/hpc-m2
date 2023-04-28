import os
import imageio.v2 as imageio
import numpy as np

subfolder = '001'
folders = os.path.join('adni_png', subfolder)

images = np.empty([4,256,170])
idx = 0
names = []
for filename in os.listdir(folders):
    if filename.endswith('.png') and '145' in filename:
        with open(os.path.join(folders, filename), 'r') as f:
            im = imageio.imread(f.name)
            names.insert(idx,f.name[-17:-4])
            images[idx,:,:] = im
            #print (names[idx], im.shape)
            idx += 1
            if idx == 4:
                break


im = images[0]
im = im -im.min() / im.max() - im.min() # normalize data 
u,s,vt = np.linalg.svd(im, full_matrices=False)


import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from workerfun import reconstruct_svd_mp, task

NP_SHARED_NAME = 'npshared21111'
NP_DATA_TYPE = np.float64
NP_SIZE = (u.shape[0], vt.shape[1])

"""
reco_shared_mem = SharedMemory(create=True, size=int(np.prod(NP_SIZE) * np.dtype(NP_DATA_TYPE).itemsize), name=NP_SHARED_NAME)
print(f"shared memory created: {reco_shared_mem.name}")
reco = np.ndarray(NP_SIZE, dtype=NP_DATA_TYPE, buffer=reco_shared_mem.buf)
print(f"shared memory ndarray created: {reco.shape}")
"""



if __name__ == '__main__':

    print("start multiprocessing")

    with mp.Pool(processes=os.cpu_count()) as pool:
        print(f"pool created with {os.cpu_count()} processes")

        # submit all jobs
        for i in range(u.shape[0]):
            pool.apply_async(task, args=(u,s,vt,30,i))
        print("all jobs submitted")

        # close the pool
        pool.close()
        print("pool closed")
        # wait for all tasks to complete
        pool.join()
        print("pool joined")


        # display results
        print("results:")
        reco_shared_mem = SharedMemory(create=True, size=int(np.prod(NP_SIZE) * np.dtype(NP_DATA_TYPE).itemsize), name=NP_SHARED_NAME)
        reco = np.ndarray(NP_SIZE, dtype=NP_DATA_TYPE, buffer=reco_shared_mem.buf)
        print(reco)
    
    # must this be created in the main process?
    reco_shared_mem.close()
    reco_shared_mem.unlink()
    """
    """
    import time
    time.sleep(2)