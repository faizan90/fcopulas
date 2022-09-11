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

cpdef np.ndarray asymmetrize_type_1_cy(
        DT_D[::1, :] data, 
        DT_D[::1, :] probs, 
        DT_UL n_levels,
        DT_D max_shift_exp,
        long long max_shift,
        DT_D pre_vals_ratio,
        DT_UL asymm_n_iters,
        DT_D[::1] asymms_rand_err) except +


cpdef np.ndarray asymmetrize_type_2_cy(
        DT_D[::1, :] data, 
        DT_D[::1, :] probs, 
        DT_UL n_levels,
        DT_D max_shift_exp,
        long long max_shift,
        DT_D pre_vals_ratio,
        DT_UL asymm_n_iters,
        DT_D[::1] asymms_rand_err,
        DT_D prob_center) except +


cpdef np.ndarray asymmetrize_type_3_cy(
        const DT_D[::1, :] data, 
        const DT_D[::1, :] probs, 
        const DT_UL n_levels,
        const DT_D max_shift_exp,
        const long long max_shift,
        const DT_D pre_vals_ratio,
        const DT_UL asymm_n_iters,
        const DT_D[::1] asymms_rand_err,
        const DT_D prob_center) except +
