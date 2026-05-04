import numpy as np
import numba
from numba import njit
import dask
from bisect import bisect_left
import multiprocessing

import dask
from numba import njit, prange, types
from numba.typed import Dict
import numpy as np

from .patches import unpack_patches

class EqualFrequencySampler:
    def __init__(self, bins, data, sample_shape, time_range_valid, time_range_sampling=None, timestep_secs=5*60, random_seed=None, preselected_samples=None):
        if time_range_sampling is None:
            time_range_sampling = time_range_valid

        # 初始化随机数生成器
        self.rng = np.random.RandomState(seed=random_seed)
        self.preselected_samples = preselected_samples

        # 从数据中生成 binned_patches
        self.binned_patches = self.generate_binned_patches(bins, *unpack_patches(data), time_range_valid, timestep_secs)

        # 初始化当前索引
        self.current_ind = np.array([len(patches) for patches in self.binned_patches])

    def generate_binned_patches(self, bins, patch,patch_times, time_range_valid, timestep_secs):
        # 使用 bin_classify_patches_parallel 函数生成 binned_patches
        zero_value = 0
        metric_func = None  # 你可以根据需要定义 metric_func
        scale = None  # 你可以根据需要定义 scale

        return bin_classify_patches_parallel(
            bins,
            patch,
            patch_times, None,  # 没有 patch_coords 和 patch_times
            None
        )

    def get_bin_sample(self, bin_ind):
        patches = self.binned_patches[bin_ind]
        sample_ind = self.current_ind[bin_ind]
        if sample_ind >= len(patches):
            self.rng.shuffle(patches)
            sample_ind = self.current_ind[bin_ind] = 0
        else:
            self.current_ind[bin_ind] += 1
        return patches[sample_ind]

    def __call__(self, num):
        # sample each bin with equal probability
        bins = self.rng.randint(len(self.binned_patches), size=num)
        coords = np.stack([self.get_bin_sample(b) for b in bins], axis=0)
        return coords

def bin_classify_patches_parallel(
    bins, data, patch_times,
    metric_func=None,
    scale=None,
):
    num_procs = multiprocessing.cpu_count()

    tasks = []
    for p in range(num_procs):
        pk0 = int(round(data.shape[0] * p / num_procs))
        pk1 = int(round(data.shape[0] * (p + 1) / num_procs))

        task = dask.delayed(bin_classify_patches)(
            bins,
            data[pk0:pk1, ...],
            patch_times[pk0:pk1],
            metric_func=metric_func,
            scale=scale
        )
        tasks.append(task)

    chunked_bins = dask.compute(tasks, scheduler="threads")[0]

    n_bins = len(chunked_bins[0])
    binned_patches = [
        np.concatenate([cb[i] for cb in chunked_bins], axis=0)
        for i in range(n_bins)
    ]
    return binned_patches

# def bin_classify_patches(
#     bins, patches, patch_times,
#     metric_func=None,
#     scale=None,
# ):
#     if metric_func is None:
#         def metric_func(x):
#             xm = np.percentile(x, 99, axis=(1, 2))
#             if np.issubdtype(x.dtype, np.integer):
#                 xm = xm.round()
#             return xm.astype(x.dtype)
#
#     binned_patches = [[] for _ in range(len(bins) + 1)]
#
#     def find_bin(value):
#         return bisect_left(bins, value)
#
#     # zero_bin = find_bin(zero_value if scale is None else scale[zero_value])
#     # for (t, (pi, pj)) in zip(zero_patch_times, zero_patch_coords):
#     #     binned_patches[zero_bin].append((t, pi, pj))
#
#     patch_metrics = metric_func(patches)
#     if scale is not None:
#         patch_metrics = scale[patch_metrics]
#     for (metric, t, (pi, pj)) in zip(patch_metrics, patch_times):
#         patch_bin = find_bin(metric)
#         binned_patches[patch_bin].append((t, pi, pj))
#
#     for i in range(len(binned_patches)):
#         if binned_patches[i]:
#             binned_patches[i] = np.array(binned_patches[i])
#         else:
#             binned_patches[i] = np.zeros((0, 3), dtype=np.int64)
#
#     return binned_patches


def bin_classify_patches(
    bins, patches, patch_times,
    metric_func=None,
    scale=None,
):
    if metric_func is None:
        def metric_func(x):
            xm = np.percentile(x, 99, axis=(1, 2))
            if np.issubdtype(x.dtype, np.integer):
                xm = xm.round()
            return xm.astype(x.dtype)

    binned_patches = [[] for _ in range(len(bins) + 1)]

    def find_bin(value):
        return bisect_left(bins, value)

    patch_metrics = metric_func(patches)
    if scale is not None:
        patch_metrics = scale[patch_metrics]

    for metric, t in zip(patch_metrics, patch_times):
        patch_bin = find_bin(metric)
        binned_patches[patch_bin].append((t,))

    for i in range(len(binned_patches)):
        if binned_patches[i]:
            binned_patches[i] = np.array(binned_patches[i])
        else:
            binned_patches[i] = np.zeros((0, 1), dtype=np.int64)

    return binned_patches


# '''