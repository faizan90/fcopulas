# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

ctypedef double double
ctypedef unsigned long DT_UL

import numpy as np
cimport numpy as np


cpdef double get_etpy_nd_from_hist(
        unsigned long long[::1] hist_nd,
        unsigned long long n_vals) except +

cpdef double get_etpy_min_nd(
        unsigned long long n_bins, unsigned long long n_dims) except +

cpdef double get_etpy_max_nd(
        unsigned long long n_bins, unsigned long long n_dims) except +

cpdef double get_etpy_nd_from_probs(
        const double[:, ::1] probs, unsigned long long n_bins) except +
