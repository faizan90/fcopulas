# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

import numpy as np
cimport numpy as np

from .misc cimport chk_uexp_ovf


cpdef get_srho_for_ecop_nd(
        const DT_D[::1] ecop, 
        unsigned long long n_dims) except +:

    cdef:
        unsigned long long i, n_cells

        DT_D cudus, scorr

    n_cells = ecop.shape[0]

    cudus = 0.0
    for i in range(n_cells):
        cudus += ecop[i]

    cudus /= <DT_D> n_cells

    scorr = (n_dims + 1.0) / (<DT_D> ((2 ** n_dims) - n_dims - 1.0))

    scorr *= ((2 ** n_dims) * cudus) - 1
    return scorr


cpdef get_srho_plus_for_hist_nd(
        const unsigned long long[::1] hist, 
        unsigned long long n_dims,
        unsigned long long n_bins,
        unsigned long long n_vals) except +:

    '''
    From: Nonparametric Measures of Multivariate Association, Nelson 1996.
    '''

    cdef:
        unsigned long long i, j, n_cells

        DT_D udcu, scorr, us_prod

        unsigned long long[::1] sclrs

    sclrs = np.zeros(n_dims, dtype=np.uint64)
    for j in range(n_dims):
        sclrs[j] = n_bins ** j

    n_cells = hist.shape[0]

    udcu = 0.0  # Integral of the product of u1...un and dC(u).
    for i in range(n_cells):

        # This makes a real difference in speed.
        if hist[i] == 0:
            continue

        us_prod = 1.0
        for j in range(n_dims):
            # Using already computed 'us' from an array resulted in a slight
            # slowdown instead of an increase in speed.
            us_prod *= (
                ((i // sclrs[j]) % n_bins) + 1.0) / <DT_D> (n_bins + 0.0)

        udcu += us_prod * (hist[i] / <DT_D> n_vals)

    scorr = (n_dims + 1.0) / (<DT_D> ((2 ** n_dims) - n_dims - 1.0))
    scorr *= ((2 ** n_dims) * udcu) - 1

    return scorr
