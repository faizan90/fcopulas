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

from .misc cimport chk_uexp_ovf, fill_hist_nd

cdef extern from "math.h" nogil:
    cdef:
        double log(double x)


cpdef double get_etpy_nd_from_hist(
        unsigned long long[::1] hist_nd,
        unsigned long long n_vals) except +:

    '''
    Shannon's entropy. Lower means higher dependence.
    '''

    cdef:
        Py_ssize_t i, n_cells

        double etpy, density

    n_cells = hist_nd.shape[0]

    etpy = 0.0
    for i in range(n_cells):
        if not hist_nd[i]:
            continue

        density = hist_nd[i] / <double> n_vals

        etpy += -density * log(density)

    return etpy


cpdef double get_etpy_min_nd(
        unsigned long long n_bins, unsigned long long n_dims) except +:

    '''
    All values in a single cell yield the lowest entropy of zero.
    '''

    cdef:
        double dens, etpy

    assert n_dims >= 2

    #dens = 1.0 / <double> n_bins

    #etpy = -log(dens)

    etpy = 0.0

    return etpy


cpdef double get_etpy_max_nd(
        unsigned long long n_bins, unsigned long long n_dims) except +:

    '''
    All values uniformly distributed yield the maximum entropy.
    '''

    cdef:
        double dens, etpy

    assert n_dims >= 2

    dens = 1.0 / <double> (n_bins ** n_dims)

    etpy = -log(dens)

    return etpy


cpdef double get_etpy_nd_from_probs(
        const double[:, ::1] probs, unsigned long long n_bins) except +:

    cdef:
        unsigned long long n_vals, n_dims

        double etpy

        unsigned long long[::1] hist_nd

    n_vals = probs.shape[0]
    n_dims = probs.shape[1]

    assert n_vals > 1, 'n_vals too low!'
    assert n_dims > 1, 'n_dims too low!'
    assert n_bins > 1, 'n_bins too low!'
    assert chk_uexp_ovf(n_bins, n_dims) == 0, (
        'Number of cells for the ecop too large!')

    hist_nd = np.zeros(n_bins ** n_dims, dtype=np.uint64)

    fill_hist_nd(probs, hist_nd, n_bins)

    etpy = get_etpy_nd_from_hist(hist_nd, n_vals)

    return etpy