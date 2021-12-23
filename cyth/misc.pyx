# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True


import numpy as np
cimport numpy as np

cdef int chk_uexp_ovf(
        const unsigned long long base,
        const unsigned long long exponent) nogil:

    cdef:
        unsigned long long i, pre_val, post_val, result

    pre_val = base
    result = 0
    for i in range(2, exponent + 1):
        post_val = base**i

        if post_val > pre_val:
            pre_val = post_val
            result = 0
        
        else:
            result = 1
            break

    return result



cdef void fill_hist_nd(
        const double[:, ::1] probs,
        unsigned long long[::1] hist_nd,
        unsigned long long n_bins) nogil except +:

    cdef:
        unsigned long long i, j, idx_temp, idx, n_vals, n_dims

    n_vals = probs.shape[0]
    n_dims = probs.shape[1]

    for i in range(n_vals):
        idx = 0
        for j in range(n_dims):
            idx_temp = <unsigned long long> (probs[i, j] * n_bins)
            idx_temp *= n_bins ** j
            idx += idx_temp

        hist_nd[idx] += 1

    return


cdef void fill_hist_nd_integral(
        const unsigned long long[::1] hist_nd,
        unsigned long long[::1] hist_nd_intg,
        unsigned long long n_dims,
        unsigned long long n_bins) except +:

    cdef:
        int sel_flag

        unsigned long long i, j, ctr, cell_intg_freq, idx, n_cells

        unsigned long long[::1] sclrs, ref_cell_idx, cur_cell_idx 

    n_cells = hist_nd.shape[0]

    sclrs = np.zeros(n_dims, dtype=np.uint64)
    for j in range(n_dims):
        sclrs[j] = n_bins ** j

    ref_cell_idx = np.empty(n_dims, dtype=np.uint64)
    cur_cell_idx = np.empty(n_dims, dtype=np.uint64)

    for i in range(n_cells):
        for j in range(n_dims):
            ref_cell_idx[j] = (<unsigned long long> (i // sclrs[j])) % n_bins

        ctr = 0
        cell_intg_freq = 0
        while ctr <= i:
            sel_flag = 1

            for j in range(n_dims):
                cur_cell_idx[j] = (
                    <unsigned long long> (ctr // sclrs[j])) % n_bins

                if cur_cell_idx[j] > ref_cell_idx[j]:
                    sel_flag = 0
                    break

            if sel_flag:
                idx = 0

                for j in range(n_dims):
                    idx += cur_cell_idx[j] * sclrs[j]

                cell_intg_freq += hist_nd[idx]    

            ctr += 1

        hist_nd_intg[i] = cell_intg_freq

    return


cdef int shift_by_one_ign(
        unsigned long long[::1] shift_idxs,
        const unsigned long long[::1] max_idxs, 
        const unsigned long long ignore_idx) nogil except +:

    cdef:
        int chg_flag = 0

        unsigned long long i, j, idx, n_dims

    n_dims = shift_idxs.shape[0]

    for i in range(n_dims):

        if shift_idxs[i] > max_idxs[i]:
            continue

        if i == ignore_idx:
            continue

        chg_flag = 1

        shift_idxs[i] += 1

        idx = i

        while True:
            if shift_idxs[idx] > max_idxs[idx]:
                for j in range(idx + 1):

                    if j == ignore_idx:
                        continue

                    shift_idxs[j] = 0

                idx += 1

                if idx == ignore_idx:
                    idx += 1

                if idx == n_dims:
                    break

                shift_idxs[idx] += 1

            else:
                break

        break

    return chg_flag


cdef void fill_hist_nd_integral_v2(
        const unsigned long long[::1] hist_nd,
        unsigned long long[::1] hist_nd_intg,
        const unsigned long long n_dims,
        const unsigned long long n_bins) except +:

    cdef:
        int take_cond

        unsigned long long i, j, k, cell_intg_freq, n_cells, idx

        unsigned long long pre_cell_idx

        unsigned long long[::1] sclrs, ref_cell_idx

        unsigned long long[::1] shift_idxs, vsted_cells, max_idxs

    n_cells = hist_nd.shape[0]

    sclrs = np.zeros(n_dims, dtype=np.uint64)
    for j in range(n_dims):
        sclrs[j] = n_bins ** j

    ref_cell_idx = np.empty(n_dims, dtype=np.uint64)

    shift_idxs = np.empty(n_dims, dtype=np.uint64)

    vsted_cells = np.empty(n_cells, dtype=np.uint64)

    max_idxs = np.empty(n_cells, dtype=np.uint64)

    for i in range(n_cells):
        pre_cell_idx = 0
        for j in range(n_dims):
            ref_cell_idx[j] = (<unsigned long long> (i // sclrs[j])) % n_bins

            if ref_cell_idx[j] <= 1:
                continue

            pre_cell_idx += (ref_cell_idx[j] - 1) * sclrs[j] 

        cell_intg_freq = hist_nd_intg[pre_cell_idx]

        for j in range(i):
            vsted_cells[j] = 0

        for j in range(n_dims):

            if not ref_cell_idx[j]:
                continue

            for k in range(n_dims):
                max_idxs[k] = ref_cell_idx[k]
                shift_idxs[k] = 0

            if (j + 1) == n_dims:
                if max_idxs[0]:
                    max_idxs[0] -= 1

                # else:
                #     continue
            else:
                if max_idxs[j + 1]:
                    max_idxs[j + 1] -= 1

                # else:
                #     continue

            shift_idxs[j] = ref_cell_idx[j]

            take_cond = 0
            for k in range(n_dims):
                if max_idxs[k] == shift_idxs[k]:
                    continue

                take_cond = 1
                break

            while take_cond:
                idx = 0
                for k in range(n_dims):
                    idx += shift_idxs[k] * sclrs[k]

                if not vsted_cells[idx]:
                    cell_intg_freq += hist_nd[idx]

                    vsted_cells[idx] += 1

                if shift_by_one_ign(shift_idxs, ref_cell_idx, j):
                    pass

                else:
                    break

                take_cond = 0
                for k in range(n_dims):
                    if ref_cell_idx[k] == shift_idxs[k]:
                        continue

                    take_cond = 1
                    break

            idx = 0
            for k in range(n_dims):
                idx += shift_idxs[k] * sclrs[k]

            if idx != i:
                if not vsted_cells[idx]:
                    cell_intg_freq += hist_nd[idx]
    
                    vsted_cells[idx] += 1

        hist_nd_intg[i] = cell_intg_freq + hist_nd[i]

    return


cpdef np.ndarray get_hist_nd(
        const double[:, ::1] probs, 
        unsigned long long n_bins) except +:

    cdef:
        unsigned long long n_vals, n_dims

        unsigned long long[::1] hist_nd

    n_vals = probs.shape[0]
    n_dims = probs.shape[1]

    assert n_vals > 1, 'n_vals too low!'
    assert n_dims > 1, 'n_dims too low!'
    assert n_bins > 1, 'n_bins too low!'
    assert chk_uexp_ovf(n_bins, n_dims) == 0, (
        'Number of cells for the ecop too large!')

    hist_nd = np.zeros(n_bins ** n_dims, dtype=np.uint64)

    fill_hist_nd(probs, hist_nd, n_bins)

    return np.asarray(hist_nd)
