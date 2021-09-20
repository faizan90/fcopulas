# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

import numpy as np
cimport numpy as np

from numpy.math cimport INFINITY

from .misc cimport chk_uexp_ovf

cdef extern from "math.h" nogil:
    cdef:
        DT_D log(DT_D x)


cpdef DT_D get_etpy_nd(
        unsigned long long[::1] hist_nd,
        unsigned long long n_vals) except +:

    '''
    Shannon's entropy. Lower means higher dependence.
    '''

    cdef:
        Py_ssize_t i, n_cells

        DT_D etpy, density

    n_cells = hist_nd.shape[0]

    etpy = 0.0
    for i in range(n_cells):
        if not hist_nd[i]:
            continue

        density = hist_nd[i] / <DT_D> n_vals

        etpy += -density * log(density)

    return etpy


cpdef DT_D get_etpy_min_nd(
        unsigned long long n_bins, unsigned long long n_dims) except +:

    '''
    All values in a single cell yield the lowest entropy of zero.
    '''

    cdef:
        DT_D dens, etpy

    assert n_dims >= 2

    dens = 1.0 / <DT_D> n_bins

    etpy = -log(dens)

#     etpy = 0.0

    return etpy


cpdef DT_D get_etpy_max_nd(
        unsigned long long n_bins, unsigned long long n_dims) except +:

    '''
    All values uniformly distributed yield the maximum entropy.
    '''

    cdef:
        DT_D dens, etpy

    assert n_dims >= 2

    dens = 1.0 / <DT_D> (n_bins ** n_dims)

    etpy = -log(dens)

    return etpy