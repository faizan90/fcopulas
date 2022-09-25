# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

from libc.stdint cimport int64_t, uint32_t, uint64_t

import numpy as np
cimport numpy as np


cpdef np.ndarray asymmetrize_type_11_ms_cy(
        const double[::1, :] data, 
        const double[::1, :] probs, 
        const uint32_t[::1] n_levelss,
        const double[::1] max_shift_exps,
        const int64_t[::1] max_shifts,
        const double[::1] pre_vals_ratios,
        const uint32_t[::1] asymm_n_iterss,
        const double[::1, :] asymms_rand_err,
        const double[::1] prob_centers,
        const double[::1] pre_val_exps,
        const double[::1] crt_val_exps,
        const int64_t[::1] level_thresh_cnsts,
        const double[::1] level_thresh_slps,
        const double[::1] rand_err_sclr_cnsts,
        const double[::1] rand_err_sclr_rels,
        const double[::1, :] rand_err_cnst,
        const double[::1, :] rand_err_rel) except +
