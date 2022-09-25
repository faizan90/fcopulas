# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

import numpy as np
cimport numpy as np

cdef extern from "math.h" nogil:
    cdef:
        double abs(double x)
        int isfinite(double x)
        double ceil(double x)

cdef extern from "complex.h" nogil:
    cdef:
        double cabs(double complex x)
        double creal(double complex x)


cdef extern from "asymmetrize_test.h" nogil:
    cdef:
        void quick_sort(
            double *arr, 
            long long first_index, 
            long long last_index)

        unsigned long long searchsorted(
            const double *arr,
            const double value,
            const unsigned long long arr_size)


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
        const double[::1, :] rand_err_rel) except +:

    cdef:
        Py_ssize_t col_idx, n_cols, step_idx, n_steps, step_idx_new
        Py_ssize_t asymm_iter

        int terminate_flag = 0

        int64_t max_shift, max_shift_level_i, shift_idx, level
        int64_t level_thresh_cnst

        uint32_t n_levels, asymm_n_iters

        double max_shift_exp, pre_vals_ratio, prob_center
        double max_shift_level_f, pre_step_value, crt_step_value
        double pre_val_exp, crt_val_exp, level_thresh_slp, level_thresh
        double rand_err_sclr_cnst, rand_err_sclr_rel

        double complex pre_val, crt_val

        int64_t[::1] data_col_levels

        double[::1] data_col_srt, data_col_asymm, data_col_asymm_level
        double[::1] data_col_asymm_tmp

        double[::1, :] out_data

    n_steps = data.shape[0]
    n_cols = data.shape[1]

    data_col_levels = np.empty(n_steps, dtype=np.int64)
    data_col_asymm = np.empty(n_steps, dtype=np.float64)
    data_col_asymm_level = np.empty(n_steps, dtype=np.float64)
    data_col_asymm_tmp = np.empty(n_steps, dtype=np.float64)
    out_data = np.empty_like(data, dtype=np.float64, order='f')
    data_col_srt = np.empty(n_steps, dtype=np.float64)

    for col_idx in range(n_cols):

        max_shift = max_shifts[col_idx]
        n_levels = <int64_t> n_levelss[col_idx]
        max_shift_exp = max_shift_exps[col_idx]
        pre_vals_ratio = pre_vals_ratios[col_idx]
        prob_center = prob_centers[col_idx]
        asymm_n_iters = asymm_n_iterss[col_idx]
        pre_val_exp = pre_val_exps[col_idx]
        crt_val_exp = crt_val_exps[col_idx]
        level_thresh_cnst = <int64_t> level_thresh_cnsts[col_idx]
        level_thresh_slp = level_thresh_slps[col_idx]
        rand_err_sclr_cnst = rand_err_sclr_cnsts[col_idx]
        rand_err_sclr_rel = rand_err_sclr_rels[col_idx]

        for step_idx in range(n_steps):
            data_col_asymm[step_idx] = data[step_idx, col_idx]

            data_col_levels[step_idx] = <int64_t> ceil(
                (abs(probs[step_idx, col_idx] - prob_center)) * n_levels)

            data_col_srt[step_idx] = data[step_idx, col_idx]

        quick_sort(&data_col_srt[0], 0, n_steps - 1)

        asymm_iter = 0
        while asymm_iter < asymm_n_iters:

            for level in range(1, n_levels + 1):

                level_thresh = level_thresh_cnst + (level_thresh_slp * level)

                for step_idx in range(n_steps):
                    data_col_asymm_level[step_idx] = data_col_asymm[step_idx]

                max_shift_level_f = <double> max_shift

                if max_shift < 0:

                    max_shift_level_f -= (
                        max_shift * 
                        ((level / <double> n_levels) ** max_shift_exp))

                    max_shift_level_i = <int64_t> (max_shift_level_f - 1.0)

                    if max_shift_level_i >= 0:
                        break

                    shift_idx = max_shift_level_i
                    while shift_idx < 0:

                        pre_step_value = data_col_asymm_level[0]
                        for step_idx in range(n_steps - 1, -1, -1):
                            crt_step_value = data_col_asymm_level[step_idx]

                            if data_col_levels[step_idx] <= level:
                                data_col_asymm_level[step_idx] = pre_step_value

                            pre_step_value = crt_step_value

                        shift_idx += 1

                elif max_shift > 0:

                    max_shift_level_f -= (
                        max_shift * 
                        ((level / <double> n_levels) ** max_shift_exp))

                    max_shift_level_i = <int64_t> (max_shift_level_f + 1.0)

                    if max_shift_level_i < 0:
                        break

                    shift_idx = 0
                    while shift_idx < max_shift_level_i:

                        pre_step_value = data_col_asymm_level[n_steps - 1]
                        for step_idx in range(n_steps):
                            crt_step_value = data_col_asymm_level[step_idx]
 
                            if data_col_levels[step_idx] <= level:
                                data_col_asymm_level[step_idx] = pre_step_value

                            pre_step_value = crt_step_value
 
                        shift_idx += 1

                for step_idx in range(n_steps):
                    if (data_col_levels[step_idx] > level) or (
                       abs(<double> (data_col_levels[step_idx] - level)) > 
                       level_thresh):

                        continue

                    pre_val = data_col_asymm[step_idx]
                    crt_val = data_col_asymm_level[step_idx]

                    # Don't refactor.
                    if creal(pre_val) < 0:

                        data_col_asymm[step_idx] = -(
                            cabs(pre_val ** pre_val_exp) * 
                            pre_vals_ratio) 

                    else:
                        data_col_asymm[step_idx] = (
                            cabs(pre_val ** pre_val_exp) * 
                            pre_vals_ratio) 

                    if creal(crt_val) < 0:

                        data_col_asymm[step_idx] -= (
                            cabs(crt_val ** crt_val_exp) * 
                            (1.0 - pre_vals_ratio))

                    else:
                        data_col_asymm[step_idx] += (
                            cabs(crt_val ** crt_val_exp) * 
                            (1.0 - pre_vals_ratio))

                    if isfinite(data_col_asymm[step_idx]) == 0:
                        # print(pre_val, crt_val, data_col_asymm[step_idx])
                        # raise Exception
                        terminate_flag = 1
                        break

                if terminate_flag:
                    break

            if terminate_flag:
                break

            for step_idx in range(n_steps):
 
                if asymm_iter < (asymm_n_iters - 1):
                    data_col_asymm[step_idx] += (
                        data_col_asymm[step_idx] * 
                        rand_err_rel[step_idx, col_idx] * 
                        rand_err_sclr_rel)

                    data_col_asymm[step_idx] += (
                        rand_err_cnst[step_idx, col_idx] * 
                        rand_err_sclr_cnst)

                data_col_asymm_tmp[step_idx] = data_col_asymm[step_idx]

            quick_sort(&data_col_asymm_tmp[0], 0, n_steps - 1)

            for step_idx in range(n_steps):
                step_idx_new = searchsorted(
                    &data_col_asymm_tmp[0], 
                    data_col_asymm[step_idx], 
                    n_steps)

                data_col_asymm[step_idx] = data_col_srt[step_idx_new]

            asymm_iter += 1

        if terminate_flag:
            break

        else:
            # Return new transformed values that do not exist in data.
            for step_idx in range(n_steps):
                out_data[step_idx, col_idx] = data_col_asymm[step_idx]

    if terminate_flag:
        for col_idx in range(n_cols):
            for step_idx in range(n_steps):
                out_data[step_idx, col_idx] = (
                    asymms_rand_err[step_idx, col_idx])

    return np.asarray(out_data)
