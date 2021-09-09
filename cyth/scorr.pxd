# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

ctypedef double DT_D
ctypedef unsigned long DT_UL

import numpy as np
cimport numpy as np


cpdef get_srho_for_ecop_nd(
        const DT_D[::1] ecop, 
        unsigned long long n_dims) except +

cpdef get_srho_plus_for_hist_nd(
        const unsigned long long[::1] hist, 
        unsigned long long n_dims,
        unsigned long long n_bins,
        unsigned long long n_vals) except +
