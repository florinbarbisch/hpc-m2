import os
import sys
from numba import cuda, float32
import math
import cupy as cp
import numpy as np
from tqdm import tqdm
import pickle
import imageio.v2 as imageio
import queue


adni_decopositions = queue.Queue()

# if load adni images
method = sys.argv[1]
NUM_STREAMS = sys.argv[2]
if method == "adni":
    for subdir, dirs, files in os.walk('adni_png'):
        print(f"Loading images from {subdir}")
        for file in tqdm(files):
            if file.endswith('.png'):
                with open(os.path.join(subdir, file), 'rb') as f:
                    image = imageio.imread(f)
                    # normalize image
                    image = image - image.min() / (image.max() - image.min())
                    # copy image to gpu
                    image_gpu = cp.asarray(image, dtype=np.float32)
                    # decompose image
                    u, s, vt = cp.linalg.svd(image_gpu)
                    # store decomposition
                    u = cp.asnumpy(u)
                    s = cp.asnumpy(s)
                    vt = cp.asnumpy(vt)
                    u_pinned = cuda.pinned_array_like(u)
                    np.copyto(u_pinned, u)
                    s_pinned = cuda.pinned_array_like(s)
                    np.copyto(s_pinned, s)
                    vt_pinned = cuda.pinned_array_like(vt)
                    np.copyto(vt_pinned, vt)

                    adni_decopositions.put({
                        "image": image,
                        "u": u_pinned,
                        "s": s_pinned,
                        "vt": vt_pinned
                    })

    print(f"Loaded {adni_decopositions.qsize()} images")
elif method == "test":
    # load decompositions from decomposition_test_image_3.pickle
    with open("decomposition_test_image_3.pickle", "rb") as f:
        decomposition = pickle.load(f)

    decomposition_pinned = {}
    # copy to pinned host memory
    decomposition_pinned["u"] = cuda.pinned_array_like(decomposition["u"])
    np.copyto(decomposition_pinned["u"], decomposition["u"])
    decomposition_pinned["s"] = cuda.pinned_array_like(decomposition["s"])
    np.copyto(decomposition_pinned["s"], decomposition["s"])
    decomposition_pinned["vt"] = cuda.pinned_array_like(decomposition["vt"])
    np.copyto(decomposition_pinned["vt"], decomposition["vt"])

    # add the test_image_3 decomposition to the queue 100 times
    for i in range(100):
        adni_decopositions.put(decomposition_pinned)
else:
    print("Please specify adni or test as first argument")
    exit(1)


threads_per_block = (4, 8)

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

if NUM_STREAMS != "serial":
    NUM_STREAMS = int(NUM_STREAMS)
    # use numba for streams and events
    streams = [cuda.stream() for _ in range(NUM_STREAMS)]
    events = [cuda.event() for _ in range(len(streams))]


    def allocate_buffers(ndarray, streams):
        shape, dtype = ndarray.shape, ndarray.dtype
        def allocate_buffer(stream):
            return cuda.device_array(shape, dtype=dtype, stream=stream)
        return ([allocate_buffer(stream) for stream in streams], [allocate_buffer(stream) for stream in streams])

    decomp = adni_decopositions.queue[0]
    gpu_u_buffers = allocate_buffers(decomp['u'], streams)
    gpu_s_buffers = allocate_buffers(decomp['s'], streams)
    gpu_vt_buffers = allocate_buffers(decomp['vt'], streams)
    gpu_C_buffers = allocate_buffers(np.zeros((decomp['u'].shape[0], decomp['vt'].shape[1]), dtype=np.float32), streams)


    # create a datastructure to remember which stream is currently accessing which buffer
    # this is needed to make sure that the buffer is not overwritten while it is still in use
    buffer_idx_in_use = [0 for _ in range(len(streams))]

    def reconstruct_one_decomposition(decomp, gpu_u_buffer, gpu_s_buffer, gpu_vt_buffer, gpu_C_buffer, threads_per_block, stream):
        gpu_u_buffer.copy_to_device(decomp['u'], stream=stream)
        gpu_s_buffer.copy_to_device(decomp['s'], stream=stream)
        gpu_vt_buffer.copy_to_device(decomp['vt'], stream=stream)
        
        # fill the C buffer with zeros
        # gpu_C_buffer[:, :] = 0.0
        k = min(gpu_u_buffer.shape[1], gpu_vt_buffer.shape[0])

        blocks_per_grid_x = math.ceil(gpu_C_buffer.shape[0] / threads_per_block[0])
        blocks_per_grid_y = math.ceil(gpu_C_buffer.shape[1] / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        reconstruct_svd_numba_shared_memory[blocks_per_grid, threads_per_block, stream](gpu_u_buffer, gpu_s_buffer, gpu_vt_buffer, gpu_C_buffer, k)
        """
        C = cuda.pinned_array_like(gpu_C_buffer)
        gpu_C_buffer.copy_to_host(C, stream=stream)
        """





    # initialize streams
    for i, (event, stream) in enumerate(zip(events, streams)):
        buf_idx = buffer_idx_in_use[i]
        reconstruct_one_decomposition(adni_decopositions.get(),
                                    gpu_u_buffers[buf_idx][i],
                                    gpu_s_buffers[buf_idx][i],
                                    gpu_vt_buffers[buf_idx][i],
                                    gpu_C_buffers[buf_idx][i],
                                    threads_per_block, stream)
        buffer_idx_in_use[i] = (buf_idx + 1) % 2
        event.record(stream=stream)

        buf_idx = buffer_idx_in_use[i]
        reconstruct_one_decomposition(adni_decopositions.get(),
                                    gpu_u_buffers[buf_idx][i],
                                    gpu_s_buffers[buf_idx][i],
                                    gpu_vt_buffers[buf_idx][i],
                                    gpu_C_buffers[buf_idx][i],
                                    threads_per_block,
                                    stream)
        buffer_idx_in_use[i] = (buf_idx + 1) % 2
    print("Initialized all streams")


    while not adni_decopositions.empty():
        for i, (event, stream) in enumerate(zip(events, streams)):
            # query events[i] if it has been recorded
            if event.query() and not adni_decopositions.empty():
                event.record(stream=stream)
                buf_idx = buffer_idx_in_use[i]
                reconstruct_one_decomposition(adni_decopositions.get(),
                                            gpu_u_buffers[buf_idx][i],
                                            gpu_s_buffers[buf_idx][i],
                                            gpu_vt_buffers[buf_idx][i],
                                            gpu_C_buffers[buf_idx][i],
                                            threads_per_block,
                                            stream)
                buffer_idx_in_use[i] = (buf_idx + 1) % 2
                print(f"Decompositions left: {adni_decopositions.qsize()}")
    print("All decompositions have been reconstructed")


    # wait for all streams to finish
    for stream in streams:
        stream.synchronize()
else:
    def reconstruct_one_decomposition_serial(adni_decopositions, threads_per_block):
        decomp = adni_decopositions.get()

        u_gpu = cuda.to_device(decomp['u'])
        s_gpu = cuda.to_device(decomp['s'])
        vt_gpu = cuda.to_device(decomp['vt'])
        C_gpu = cuda.device_array((u_gpu.shape[0], vt_gpu.shape[1]), dtype=np.float32)
        k = min(u_gpu.shape[1], vt_gpu.shape[0])

        blocks_per_grid_x = math.ceil(C_gpu.shape[0] / threads_per_block[0])
        blocks_per_grid_y = math.ceil(C_gpu.shape[1] / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        reconstruct_svd_numba_shared_memory[blocks_per_grid, threads_per_block](u_gpu, s_gpu, vt_gpu, C_gpu, k)

    while not adni_decopositions.empty():
        reconstruct_one_decomposition_serial(adni_decopositions, threads_per_block)
        print(f"Decompositions left: {adni_decopositions.qsize()}")