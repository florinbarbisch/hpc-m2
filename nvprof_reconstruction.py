import math
import time
import sys
import pickle
import cupy as cp
from numba import cuda, float32

# parse arguments
threads_per_block = (int(sys.argv[1]), int(sys.argv[2]))

# load decompositions from decomposition_test_image_3.pickle
with open("decomposition_test_image_3.pickle", "rb") as f:
    decompositions = pickle.load(f)
    u = decompositions["u"]
    s = decompositions["s"]
    vt = decompositions["vt"]

print("Image size: {}x{}".format(u.shape[0], vt.shape[1]))

u_gpu, s_gpu, vt_gpu = cp.array(u, dtype=cp.float32), cp.array(s, dtype=cp.float32), cp.array(vt, dtype=cp.float32)

print("SVD shapes: {}, {}, {}".format(u_gpu.shape, s_gpu.shape, vt_gpu.shape))

k = min(u_gpu.shape[1], vt_gpu.shape[0]) // 3 # use 1/3 of components


@cuda.jit
def reconstruct_svd_numba_shared_memory(u, s, vt, C, k):
    """
    This kernel uses shared memory to speed up the computation.
    
    :param u: u matrix
    :param s: s vector
    :param vt: vt matrix
    :param C: result matrix
    :param k: number of components to keep
    """
    block_i = cuda.blockIdx.x
    block_j = cuda.blockIdx.y
    thread_i = cuda.threadIdx.x
    thread_j = cuda.threadIdx.y
    i, j = cuda.grid(2)

    tmp = 0.0
    
    u_shared = cuda.shared.array(shape=(threads_per_block[0], threads_per_block[1]), dtype=float32)
    vt_shared = cuda.shared.array(shape=(threads_per_block[0], threads_per_block[1]), dtype=float32)
    s_shared = cuda.shared.array(shape=(threads_per_block[0]), dtype=float32)

    num_blocks = math.ceil(min(k, vt.shape[0], u.shape[1]) / threads_per_block[0])
    for m in range(num_blocks):
        u_shared[thread_i, thread_j] = u[block_i * threads_per_block[0] + thread_i, m * threads_per_block[1] + thread_j]
        vt_shared[thread_i, thread_j] = vt[m * threads_per_block[0] + thread_i, block_j * threads_per_block[1] + thread_j]
        if thread_j == 0:
            s_shared[thread_i] = s[m * threads_per_block[0] + thread_i]

        cuda.syncthreads()
        for l in range(threads_per_block[0]):
            if l + m * threads_per_block[0] < k:
                tmp += u_shared[thread_i, l] * s_shared[l] * vt_shared[l, thread_j]
        cuda.syncthreads()

    C[i, j] = tmp

blocks_per_grid_x = math.ceil(u_gpu.shape[0] / threads_per_block[0])
blocks_per_grid_y = math.ceil(vt_gpu.shape[1] / threads_per_block[1])
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

print("Blocks per grid: {}".format(blocks_per_grid))

C_gpu = cp.zeros((u_gpu.shape[0], vt_gpu.shape[1]), dtype=cp.float32)

start = time.perf_counter()
reconstruct_svd_numba_shared_memory[blocks_per_grid, threads_per_block](u_gpu, s_gpu, vt_gpu, C_gpu, k)

print(f"Done in {time.perf_counter() - start} seconds!")