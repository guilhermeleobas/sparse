import numpy as np
import scipy.sparse as sps
from numba import typed, njit
from numba.core.types import float64
from typing import List, Tuple
from sparse._meta.dense_level import Dense
from sparse._meta.compressed_level import Compressed
from sparse._meta.format import Tensor, Format

@njit
def matvec_mul_kernel(
    m: Tuple[Dense, Compressed],
    v: Tuple[Compressed],
    m_data: typed.List,
    v_data: typed.List,
    shape
):
    out = (Dense(N=shape[1], ordered=True, unique=True), np.zeros((1024,), dtype=np.float64))
    out[0].insert_init(1, shape[1])
    p1begin, p1end = m[0].coord_bounds(())
    for i in range(p1begin, p1end):
        p1, _ = m[0].coord_access(0, (0,))
        val = float64(0)
        p2begin, p2end = m[1].pos_bounds(p1)
        for jA in range(p2begin, p2end):
            j, found = m[1].pos_access(jA, (i,))
            val += m_data[jA] * v_data[j]
        out[0].insert_coord(i, i)
        # out[1].append(val)
    out[0].insert_finalize(1, shape[1])
    return out

def generate_random_data(shape : Tuple[int, int], seed : int, density=0.01) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    size = np.prod(shape)
    nnz = int(density * size)
    rng = np.random.Generator(np.random.MT19937(seed=seed))
    linear_idxs = rng.choice(size, size=nnz, replace=False)
    data_2d = rng.normal(size=nnz)
    data_1d = rng.normal(size=shape[1])
    multi_idx = np.unravel_index(linear_idxs, shape=shape)
    coords_2d = tuple(np.array(multi_idx).T)
    return multi_idx, data_1d, data_2d
    
def create_tensors(shape: Tuple[int, int], multi_idx: List[Tuple[int, int]], data_2d: List[int], data_1d: List[int]):
    csr_matrix = sps.csr_matrix((data_2d, tuple(multi_idx)), shape=shape)
    coords_2d = tuple(np.array(multi_idx).T)
    m = Tensor(shape=shape, fmt=Format(name='csr', levels=(Dense, Compressed)))
    v = Tensor(shape=(shape[1],), fmt=Format(name='dense_vector', levels=(Dense,)))
    m.insert_data(coords=coords_2d, data=data_2d)
    coords_1d = tuple(np.arange(shape[1])[None, :])
    v.insert_data(coords=coords_1d, data=data_1d)
    return csr_matrix, data_1d, m, v


class SpMVSuite:
    def setup(self):
        shape = (1000, 1000)
        multi_idx, data_1d, data_2d = generate_random_data(shape, seed=1337)
        self.sp_m, self.sp_v, self.m, self.v = create_tensors(shape, multi_idx, data_2d, data_1d)
        self.shape = shape
        matvec_mul_kernel(self.m.levels, self.v.levels, self.m.data, self.v.data, shape)
    
    def bench_scipy_mv(self):
        self.sp_m @ self.sp_v
    
    def bench_taco(self):
        matvec_mul_kernel(self.m.levels, self.v.levels, self.m.data, self.v.data, self.shape)
