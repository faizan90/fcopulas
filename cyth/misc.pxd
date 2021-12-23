# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

import numpy as np
cimport numpy as np

ctypedef unsigned long DT_UL


cdef int chk_uexp_ovf(
        const unsigned long long base,
        const unsigned long long exponent) nogil

cdef void fill_hist_nd(
        const double[:, ::1] probs,
        unsigned long long[::1] hist_nd,
        unsigned long long n_bins) nogil except +

cdef void fill_hist_nd_integral(
        const unsigned long long[::1] hist_nd,
        unsigned long long[::1] hist_nd_intg,
        unsigned long long n_dims,
        unsigned long long n_bins) except +

cdef void fill_hist_nd_integral_v2(
        const unsigned long long[::1] hist_nd,
        unsigned long long[::1] hist_nd_intg,
        const unsigned long long n_dims,
        const unsigned long long n_bins) except +

cpdef np.ndarray get_hist_nd(
        const double[:, ::1] probs, 
        unsigned long long n_bins) except +
