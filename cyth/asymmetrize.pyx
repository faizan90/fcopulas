# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

# from libc.math import round

import numpy as np
cimport numpy as np

cdef extern from "math.h" nogil:
    cdef:
        DT_D abs(DT_D x)

cdef extern from "complex.h" nogil:
    cdef:
        DT_D cabs(double complex x)


cdef extern from "asymmetrize.h" nogil:
    cdef:
        void quick_sort(
            double *arr, 
            long long first_index, 
            long long last_index)

        unsigned long long searchsorted(
            const double *arr,
            const double value,
            const unsigned long long arr_size)


cpdef np.ndarray asymmetrize_type_1_cy(
        DT_D[::1, :] data, 
        DT_D[::1, :] probs, 
        DT_UL n_levels,
        DT_D max_shift_exp,
        long long max_shift,
        DT_D pre_vals_ratio,
        DT_UL asymm_n_iters,
        DT_D[::1] asymms_rand_err) except +:

    cdef:
        Py_ssize_t col_idx, n_cols, step_idx, n_steps, shift_idx, step_idx_new
        Py_ssize_t level, shift_idx_b, asymm_iter

        long long max_shift_level_i

        DT_D max_shift_level_f, pre_step_value, crt_step_value

        DT_D pre_vals_ratio_b

        unsigned long long[::1] data_col_levels


        DT_D[::1] data_col_srt, data_col_asymm, data_col_asymm_level
        DT_D[::1] data_col_asymm_srt, data_col_asymm_tmp

        DT_D[::1, :] out_data

    n_steps = data.shape[0]
    n_cols = data.shape[1]

    pre_vals_ratio_b = (1.0 - pre_vals_ratio)

    data_col_levels = np.empty(n_steps, dtype=np.uint64)

    data_col_srt = np.empty(n_steps, dtype=np.float64)
    data_col_asymm = np.empty(n_steps, dtype=np.float64)
    data_col_asymm_level = np.empty(n_steps, dtype=np.float64)
    data_col_asymm_tmp = np.empty(n_steps, dtype=np.float64)

    out_data = np.empty_like(data, dtype=np.float64, order='f')

    for col_idx in range(n_cols):

        for step_idx in range(n_steps):
            data_col_asymm[step_idx] = data[step_idx, col_idx]

            data_col_levels[step_idx] = <DT_UL> (
                probs[step_idx, col_idx] * n_levels)

            data_col_srt[step_idx] = data[step_idx, col_idx]

        quick_sort(&data_col_srt[0], 0, n_steps - 1)

        for asymm_iter in range(asymm_n_iters):

            for level in range(n_levels):
                for step_idx in range(n_steps):
                    data_col_asymm_level[step_idx] = data_col_asymm[step_idx]

                max_shift_level_f = <DT_D> max_shift

                max_shift_level_f -= (
                    <DT_D> max_shift * cabs(
                        (<DT_D> level / <DT_D> n_levels) ** max_shift_exp))

                max_shift_level_i = <long long> max_shift_level_f

                if max_shift_level_i == 0:
                    break

                elif max_shift_level_i < 0:
                    for shift_idx in range(max_shift_level_i, 0, 1):
                        for shift_idx_b in range(shift_idx, 0, 1):
                            pre_step_value = data_col_asymm_level[0]
                            for step_idx in range(n_steps - 1, -1, -1):
                                crt_step_value = data_col_asymm_level[step_idx]
                                data_col_asymm_level[step_idx] = pre_step_value
                                pre_step_value = crt_step_value

                        for step_idx in range(n_steps):
                            if data_col_levels[step_idx] > level:
                                continue

                            data_col_asymm_tmp[step_idx] = (
                                data_col_asymm[step_idx] * pre_vals_ratio) 

                            data_col_asymm_tmp[step_idx] += (  
                                data_col_asymm_level[step_idx] * pre_vals_ratio_b)

                        for step_idx in range(n_steps):
                            if data_col_levels[step_idx] > level:
                                continue

                            data_col_asymm[step_idx] = data_col_asymm_tmp[step_idx]

                elif max_shift_level_i > 0:

                    for shift_idx in range(max_shift_level_i):
                        for shift_idx_b in range(shift_idx + 1):
                            pre_step_value = data_col_asymm_level[n_steps - 1]
                            for step_idx in range(n_steps):
                                crt_step_value = data_col_asymm_level[step_idx]
                                data_col_asymm_level[step_idx] = pre_step_value
                                pre_step_value = crt_step_value

                        for step_idx in range(n_steps):
                            if data_col_levels[step_idx] > level:
                                continue

                            # data_col_asymm_tmp[step_idx] = (
                            #     data_col_asymm_level[step_idx]) 

                            data_col_asymm_tmp[step_idx] = (
                                data_col_asymm[step_idx] * pre_vals_ratio) 

                            data_col_asymm_tmp[step_idx] += (  
                                data_col_asymm_level[step_idx] * pre_vals_ratio_b)

                        for step_idx in range(n_steps):
                            if data_col_levels[step_idx] > level:
                                continue

                            data_col_asymm[step_idx] = data_col_asymm_tmp[step_idx]

        # # This makes it even smoother. Ideally, there should be a flag
        # # for this. It works for discharge but maybe not for precipitation.
        # # A smoothing function can be implemented seperately.
        # for step_idx in range(n_steps):
        #     data_col_asymm_tmp[step_idx] = data_col_asymm[step_idx]
        #
        # pre_step_value = data_col_asymm_tmp[n_steps - 1]
        # for step_idx in range(n_steps):
        #     crt_step_value = data_col_asymm_tmp[step_idx]
        #
        #     data_col_asymm[step_idx] = (
        #         (pre_step_value * pre_vals_ratio) + 
        #         (crt_step_value * pre_vals_ratio_b))
        #
        #     pre_step_value = crt_step_value

        for step_idx in range(n_steps):
            data_col_asymm[step_idx] += asymms_rand_err[step_idx]
            data_col_asymm_tmp[step_idx] = data_col_asymm[step_idx]

        quick_sort(&data_col_asymm_tmp[0], 0, n_steps - 1)

        for step_idx in range(n_steps):
            step_idx_new = searchsorted(
                &data_col_asymm_tmp[0], data_col_asymm[step_idx], n_steps)

            out_data[step_idx, col_idx] = data_col_srt[step_idx_new]

    return np.asarray(out_data)


cpdef np.ndarray asymmetrize_type_2_cy(
        DT_D[::1, :] data, 
        DT_D[::1, :] probs, 
        DT_UL n_levels,
        DT_D max_shift_exp,
        long long max_shift,
        DT_D pre_vals_ratio,
        DT_UL asymm_n_iters,
        DT_D[::1] asymms_rand_err,
        DT_D prob_center) except +:

    cdef:
        Py_ssize_t col_idx, n_cols, step_idx, n_steps, shift_idx, step_idx_new
        Py_ssize_t level, shift_idx_b, asymm_iter

        long long max_shift_level_i

        DT_D max_shift_level_f, pre_step_value, crt_step_value

        DT_D pre_vals_ratio_b

        unsigned long long[::1] data_col_levels


        DT_D[::1] data_col_srt, data_col_asymm, data_col_asymm_level
        DT_D[::1] data_col_asymm_srt, data_col_asymm_tmp

        DT_D[::1, :] out_data

    n_steps = data.shape[0]
    n_cols = data.shape[1]

    pre_vals_ratio_b = (1.0 - pre_vals_ratio)

    data_col_levels = np.empty(n_steps, dtype=np.uint64)

    data_col_srt = np.empty(n_steps, dtype=np.float64)
    data_col_asymm = np.empty(n_steps, dtype=np.float64)
    data_col_asymm_level = np.empty(n_steps, dtype=np.float64)
    data_col_asymm_tmp = np.empty(n_steps, dtype=np.float64)

    out_data = np.empty_like(data, dtype=np.float64, order='f')

    for col_idx in range(n_cols):

        for step_idx in range(n_steps):
            data_col_asymm[step_idx] = data[step_idx, col_idx]

            data_col_levels[step_idx] = <DT_UL> (
                (abs(probs[step_idx, col_idx] - 
                     prob_center) / (1 - prob_center)) * n_levels)

            data_col_srt[step_idx] = data[step_idx, col_idx]

        quick_sort(&data_col_srt[0], 0, n_steps - 1)

        for asymm_iter in range(asymm_n_iters):

            for level in range(n_levels):
                for step_idx in range(n_steps):
                    data_col_asymm_level[step_idx] = data_col_asymm[step_idx]

                max_shift_level_f = <DT_D> max_shift

                max_shift_level_f -= (
                    <DT_D> max_shift * cabs(
                        (<DT_D> level / <DT_D> n_levels) ** max_shift_exp))

                max_shift_level_i = <long long> max_shift_level_f

                if max_shift_level_i == 0:
                    break

                elif max_shift_level_i < 0:
                    for shift_idx in range(max_shift_level_i, 0, 1):
                        for shift_idx_b in range(shift_idx, 0, 1):
                            pre_step_value = data_col_asymm_level[0]
                            for step_idx in range(n_steps - 1, -1, -1):
                                crt_step_value = data_col_asymm_level[step_idx]
                                data_col_asymm_level[step_idx] = pre_step_value
                                pre_step_value = crt_step_value

                        for step_idx in range(n_steps):
                            if data_col_levels[step_idx] > level:
                                continue

                            data_col_asymm_tmp[step_idx] = (
                                data_col_asymm[step_idx] * pre_vals_ratio) 

                            data_col_asymm_tmp[step_idx] += (  
                                data_col_asymm_level[step_idx] * pre_vals_ratio_b)

                        for step_idx in range(n_steps):
                            if data_col_levels[step_idx] > level:
                                continue

                            data_col_asymm[step_idx] = data_col_asymm_tmp[step_idx]

                elif max_shift_level_i > 0:
                    for shift_idx in range(max_shift_level_i):
                        for shift_idx_b in range(shift_idx + 1):
                            pre_step_value = data_col_asymm_level[n_steps - 1]
                            for step_idx in range(n_steps):
                                crt_step_value = data_col_asymm_level[step_idx]
                                data_col_asymm_level[step_idx] = pre_step_value
                                pre_step_value = crt_step_value

                    for step_idx in range(n_steps):
                        if data_col_levels[step_idx] > level:
                            continue

                        # data_col_asymm_tmp[step_idx] = (
                        #     data_col_asymm_level[step_idx]) 

                        data_col_asymm_tmp[step_idx] = (
                            data_col_asymm[step_idx] * pre_vals_ratio) 

                        data_col_asymm_tmp[step_idx] += (  
                            data_col_asymm_level[step_idx] * pre_vals_ratio_b)

                    for step_idx in range(n_steps):
                        if data_col_levels[step_idx] > level:
                            continue

                        data_col_asymm[step_idx] = data_col_asymm_tmp[step_idx]

        # This makes it even smoother. Ideally, there should be a flag
        # for this. It works for discharge but maybe not for precipitation.
        # A smoothing function can be implemented seperately.
        for step_idx in range(n_steps):
            data_col_asymm_tmp[step_idx] = data_col_asymm[step_idx]

        pre_step_value = data_col_asymm_tmp[n_steps - 1]
        for step_idx in range(n_steps):
            crt_step_value = data_col_asymm_tmp[step_idx]

            data_col_asymm[step_idx] = (
                (pre_step_value * pre_vals_ratio) + 
                (crt_step_value * pre_vals_ratio_b))

            pre_step_value = crt_step_value

        for step_idx in range(n_steps):
            data_col_asymm[step_idx] += asymms_rand_err[step_idx]
            data_col_asymm_tmp[step_idx] = data_col_asymm[step_idx]

        quick_sort(&data_col_asymm_tmp[0], 0, n_steps - 1)

        for step_idx in range(n_steps):
            step_idx_new = searchsorted(
                &data_col_asymm_tmp[0], data_col_asymm[step_idx], n_steps)

            out_data[step_idx, col_idx] = data_col_srt[step_idx_new]

    return np.asarray(out_data)


cpdef np.ndarray asymmetrize_type_3_cy(
        const DT_D[::1, :] data, 
        const DT_D[::1, :] probs, 
        const DT_UL n_levels,
        const DT_D max_shift_exp,
        const long long max_shift,
        const DT_D pre_vals_ratio,
        const DT_UL asymm_n_iters,
        const DT_D[::1] asymms_rand_err,
        const DT_D prob_center) except +:

    cdef:
        Py_ssize_t col_idx, n_cols, step_idx, n_steps, shift_idx, step_idx_new
        Py_ssize_t level, shift_idx_b, asymm_iter

        long long max_shift_level_i

        DT_D max_shift_level_f, pre_step_value, crt_step_value

        DT_D pre_vals_ratio_b

        unsigned long long[::1] data_col_levels

        DT_D[::1] data_col_srt, data_col_asymm, data_col_asymm_level
        DT_D[::1] data_col_asymm_srt, data_col_asymm_tmp

        DT_D[::1, :] out_data

    n_steps = data.shape[0]
    n_cols = data.shape[1]

    pre_vals_ratio_b = (1.0 - pre_vals_ratio)

    data_col_levels = np.empty(n_steps, dtype=np.uint64)

    data_col_srt = np.empty(n_steps, dtype=np.float64)
    data_col_asymm = np.empty(n_steps, dtype=np.float64)
    data_col_asymm_level = np.empty(n_steps, dtype=np.float64)
    data_col_asymm_tmp = np.empty(n_steps, dtype=np.float64)

    out_data = np.empty_like(data, dtype=np.float64, order='f')

    for col_idx in range(n_cols):

        for step_idx in range(n_steps):
            data_col_asymm[step_idx] = data[step_idx, col_idx]

            # data_col_levels[step_idx] = <DT_UL> (
            #     ((abs(probs[step_idx, col_idx] - 
            #          prob_center) / (1 - prob_center)) ** max_shift_exp) * n_levels)

            # data_col_levels[step_idx] = <DT_UL> (
            #     (abs(probs[step_idx, col_idx] - 
            #          prob_center) / (1 - prob_center)) * n_levels)

            data_col_levels[step_idx] = <DT_UL> (
                (abs(probs[step_idx, col_idx] - prob_center)) * n_levels)

            data_col_srt[step_idx] = data[step_idx, col_idx]

        quick_sort(&data_col_srt[0], 0, n_steps - 1)

        for asymm_iter in range(asymm_n_iters):

            for level in range(n_levels):
                for step_idx in range(n_steps):
                    data_col_asymm_level[step_idx] = data_col_asymm[step_idx]

                max_shift_level_f = <DT_D> max_shift

                if max_shift < 0:
                    # max_shift_level_f -= (
                    #     <DT_D> max_shift * 
                    #     cabs(<DT_D> level / <DT_D> n_levels))

                    max_shift_level_f -= (
                        <DT_D> max_shift * 
                        (cabs(<DT_D> level / <DT_D> n_levels) ** max_shift_exp))

                    max_shift_level_i = <long long> (max_shift_level_f - 1)

                    for shift_idx in range(max_shift_level_i, 0, 1):
                        pre_step_value = data_col_asymm_level[0]
                        for step_idx in range(n_steps - 1, -1, -1):
                            crt_step_value = data_col_asymm_level[step_idx]

                            if data_col_levels[step_idx] < level:
                                data_col_asymm_level[step_idx] = pre_step_value

                            pre_step_value = crt_step_value

                    for step_idx in range(n_steps):
                        if data_col_levels[step_idx] >= level:
                            continue

                        data_col_asymm_tmp[step_idx] = (
                            data_col_asymm[step_idx] * pre_vals_ratio) 

                        data_col_asymm_tmp[step_idx] += (  
                            data_col_asymm_level[step_idx] * pre_vals_ratio_b)

                    for step_idx in range(n_steps):
                        if data_col_levels[step_idx] >= level:
                            continue

                        data_col_asymm[step_idx] = data_col_asymm_tmp[step_idx]

                elif max_shift > 0:
                    max_shift_level_f -= (
                        <DT_D> max_shift * 
                        (cabs(<DT_D> level / <DT_D> n_levels) ** max_shift_exp))

                    max_shift_level_i = <long long> (max_shift_level_f + 1.0)

                    for shift_idx in range(0, max_shift_level_i, 1):
                        pre_step_value = data_col_asymm_level[n_steps - 1]
                        for step_idx in range(n_steps):
                            crt_step_value = data_col_asymm_level[step_idx]

                            if data_col_levels[step_idx] < level:
                                data_col_asymm_level[step_idx] = pre_step_value

                            pre_step_value = crt_step_value

                    for step_idx in range(n_steps):
                        if data_col_levels[step_idx] >= level:
                            continue

                        data_col_asymm_tmp[step_idx] = (
                            data_col_asymm[step_idx] * pre_vals_ratio) 

                        data_col_asymm_tmp[step_idx] += (  
                            data_col_asymm_level[step_idx] * pre_vals_ratio_b)

                    for step_idx in range(n_steps):
                        if data_col_levels[step_idx] >= level:
                            continue

                        data_col_asymm[step_idx] = data_col_asymm_tmp[step_idx]

        # This makes it even smoother. Ideally, there should be a flag
        # for this. It works for discharge but maybe not for precipitation.
        # A smoothing function can be implemented seperately.
        # for step_idx in range(n_steps):
        #     data_col_asymm_tmp[step_idx] = data_col_asymm[step_idx]
        #
        # pre_step_value = data_col_asymm_tmp[n_steps - 1]
        # for step_idx in range(n_steps):
        #     crt_step_value = data_col_asymm_tmp[step_idx]
        #
        #     data_col_asymm[step_idx] = (
        #         (pre_step_value * pre_vals_ratio) + 
        #         (crt_step_value * pre_vals_ratio_b))
        #
        #     pre_step_value = crt_step_value

        for step_idx in range(n_steps):
            data_col_asymm[step_idx] += asymms_rand_err[step_idx]
            data_col_asymm_tmp[step_idx] = data_col_asymm[step_idx]
        
        quick_sort(&data_col_asymm_tmp[0], 0, n_steps - 1)
        
        for step_idx in range(n_steps):
            step_idx_new = searchsorted(
                &data_col_asymm_tmp[0], data_col_asymm[step_idx], n_steps)
        
            out_data[step_idx, col_idx] = data_col_srt[step_idx_new]

        # for step_idx in range(n_steps):
        #     out_data[step_idx, col_idx] = (
        #         data_col_asymm[step_idx] + asymms_rand_err[step_idx])

    return np.asarray(out_data)
