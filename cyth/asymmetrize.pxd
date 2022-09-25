# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

ctypedef double DT_D
ctypedef unsigned long DT_UL

from libc.stdint cimport int64_t, uint32_t, uint64_t

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

cpdef np.ndarray asymmetrize_type_3_ms_cy(
        const double[::1, :] data, 
        const double[::1, :] probs, 
        const uint32_t[::1] n_levelss,
        const double[::1] max_shift_exps,
        const int64_t[::1] max_shifts,
        const double[::1] pre_vals_ratios,
        const uint32_t[::1] asymm_n_iterss,
        const double[::1] asymms_rand_err,
        const double[::1] prob_centers) except +

cpdef np.ndarray asymmetrize_type_4_ms_cy(
        const double[::1, :] data, 
        const double[::1, :] probs, 
        const uint32_t[::1] n_levelss,
        const double[::1] max_shift_exps,
        const int64_t[::1] max_shifts,
        const double[::1] pre_vals_ratios,
        const uint32_t[::1] asymm_n_iterss,
        const double[::1] asymms_rand_err,
        const double[::1] prob_centers) except +

cpdef np.ndarray asymmetrize_type_5_ms_cy(
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
        const double[::1] crt_val_exps) except +

cpdef np.ndarray asymmetrize_type_6_ms_cy(
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
        const double[::1] crt_val_exps) except +

cpdef np.ndarray asymmetrize_type_7_ms_cy(
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
        const double[::1] level_thresh_slps) except +

cpdef np.ndarray asymmetrize_type_8_ms_cy(
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
        const double[::1] level_thresh_slps) except +

cpdef np.ndarray asymmetrize_type_9_ms_cy(
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
        const double[::1] rand_err_sclrs) except +

cpdef np.ndarray asymmetrize_type_10_ms_cy(
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
        