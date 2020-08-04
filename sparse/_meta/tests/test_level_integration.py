import numpy as np

from numba import njit, typed
from numba.core.types import int64, float64

from sparse._meta.compressed_level import Compressed
from sparse._meta.dense_level import Dense
from sparse._meta.format import Format, Tensor
from typing import Tuple, List
import random

nb_list = typed.List.empty_list

# @njit
def matvec_mul_kernel(
    m: Tuple[Dense, Compressed],
    v: Tuple[Compressed],
    m_data: typed.List,
    v_data: typed.List,
    shape
):
    out = (Dense(N=shape[1], ordered=True, unique=True), typed.List.empty_list(float64))
    out[0].insert_init(1, shape[1])
    for pk1, i in m[0].iterate(0, ()):
        val = float64(0)
        for jA, j in m[1].iterate(i, (i,)):
            val += m_data[jA] * v_data[j]
        out[0].insert_coord(i, i)
        out[1].append(val)
    out[0].insert_finalize(1, shape[1])
    return out
    

def matmul_kernel(
    m : Tuple[Dense, Compressed], 
    n : Tuple[Dense, Compressed], 
    m_data = typed.List,
    n_data = typed.List,
    shape = None
):
    pass


def generate_random_data(shape : Tuple[int, int], seed : int, density=0.01) -> Tuple[List[Tuple[int, int]], List[float]]:
    size = np.prod(shape)
    nnz = int(density * size)
    rng = np.random.Generator(np.random.MT19937(seed=1337))
    linear_idxs = rng.choice(size, size=nnz, replace=False)
    data1 = rng.normal(size=nnz)
    data2 = rng.normal(size=shape[1])
    multi_idx = np.unravel_index(linear_idxs, shape=shape)
    coords1 = tuple(np.array(multi_idx).T)
    m = Tensor(shape=shape, fmt=Format(name='csr', levels=(Dense, Compressed)))
    v = Tensor(shape=(shape[1],), fmt=Format(name='dense_vector', levels=(Dense,)))
    m.insert_data(coords=coords1, data=data1)
    coords2 = tuple(np.arange(shape[1])[None, :])
    v.insert_data(coords=coords2, data=data2)
    return m, v
    

def test_benchmark():
    shape = (7, 7)
    coords_2d = [(2, 3), (5, 1), (5, 3), (6, 0), (6, 4)]
    coords_1d = [(0,), (1,), (2,), (3,), (4,), (5,), (6,)]
    data_2d = [1.0, 2.0, 3.0, 4.0, 5.0]
    data_1d = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    csr = Format(name='csr', levels=(Dense, Compressed))
    m = Tensor(shape=shape, fmt=csr)    
    m.insert_data(coords=coords_2d, data=data_2d)

    dense = Format(name='dv', levels=(Dense,))
    v = Tensor(shape=(shape[1],), fmt=dense)
    v.insert_data(coords=coords_1d, data=data_1d)

    out = matvec_mul_kernel(m._levels, v._levels, m.data, v.data, shape)
    assert list(out[1]) == [0., 0., 4., 0., 0., 16., 29.]


@njit
def mul_sparse_dense_1d(
    c: Compressed,
    c_data: typed.List,
    d: Dense,
    d_data: typed.List,
    out: Compressed,
    data_out: typed.List,
):
    out.append_init(1, 100)
    pos_iter = c.pos_iter(0)
    pbeginc = 0
    for p_c1 in pos_iter:
        i_c1, found_c1 = c.pos_access(p_c1, ())
        p_d1, found_d1 = d.locate(0, (i_c1,))
        if found_c1 and found_d1:
            out.append_coord(p_c1, i_c1)
            data_c = c_data[p_c1]
            data_d = d_data[p_d1]
            data_out.append(data_c * data_d)
    pendc = p_c1 + 1
    out.append_edges(0, pbeginc, pendc)
    out.append_finalize(1, 100)


def test_sparse_dense_mul_1d():
    pos = typed.List([0, 3])
    crd = typed.List([0, 7, 39])
    c = Compressed(full=True, ordered=True, unique=True, pos=pos, crd=crd)
    data_c = typed.List([types.int64(e) for e in [7, 13, 29]])

    d = Dense(N=100, unique=True, ordered=True)
    data_d = typed.List([types.int64(e) for e in range(100)])

    out = Compressed(
        full=True,
        ordered=True,
        unique=True,
        pos=typed.List.empty_list(types.int64),
        crd=typed.List.empty_list(types.int64),
    )
    data_out = typed.List.empty_list(types.int64)
    data_expected = typed.List([7 * 0, 13 * 7, 29 * 39])

    mul_sparse_dense_1d(c, data_c, d, data_d, out, data_out)
    assert c.pos == out.pos
    assert c.crd == out.crd
    assert data_expected == data_out
