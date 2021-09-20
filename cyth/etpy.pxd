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


cpdef DT_D get_etpy_nd(
        unsigned long long[::1] hist_nd,
        unsigned long long n_vals) except +

cpdef DT_D get_etpy_min_nd(
        unsigned long long n_bins, unsigned long long n_dims) except +

cpdef DT_D get_etpy_max_nd(
        unsigned long long n_bins, unsigned long long n_dims) except +
