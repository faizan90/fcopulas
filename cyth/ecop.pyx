# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

import numpy as np
cimport numpy as np

from numpy.math cimport INFINITY

from .misc cimport chk_uexp_ovf

cdef extern from "math.h" nogil:
    cdef:
        DT_D log(DT_D x)


cdef extern from "./rand_gen_mp.h" nogil:
    cdef:
        DT_D rand_c()
        void warm_up()  # Call this everytime.

warm_up()


cpdef void fill_bin_idxs_ts(
        const DT_D[::1] probs,
              DT_UL[::1] bins_ts, 
        const DT_UL n_bins) except +:

    cdef:
        Py_ssize_t i, n_vals = probs.shape[0]

    assert bins_ts.shape[0] == n_vals
    assert n_vals >= n_bins

    for i in range(n_vals):    
        bins_ts[i] = 0

    for i in range(n_vals):    
        bins_ts[i] = <DT_UL> (n_bins * probs[i])

    return


cpdef void fill_bin_dens_1d(
        const DT_UL[::1] bins_ts,
              DT_D[::1] bins_dens) except +:
    
    cdef:
        Py_ssize_t i
        DT_UL n_bins = bins_dens.shape[0]
        DT_UL n_vals = bins_ts.shape[0]

    for i in range(n_bins):
        bins_dens[i] = 0

    # Frequencies.
    for i in range(bins_ts.shape[0]):
        bins_dens[bins_ts[i]] += 1

    # Densities.
    for i in range(n_bins):
        bins_dens[i] /= n_vals

    return


cpdef void fill_bin_dens_2d(
        const DT_UL[::1] bins_ts_x,
        const DT_UL[::1] bins_ts_y,
              DT_D[:, ::1] bins_dens_xy) except +:

    cdef:
        Py_ssize_t i, j
        DT_UL n_vals = bins_ts_x.shape[0], n_bins = bins_dens_xy.shape[0]

    assert bins_dens_xy.shape[1] == n_bins
    assert n_vals == bins_ts_y.shape[0]

    for i in range(n_bins):
        for j in range(n_bins):
            bins_dens_xy[i, j] = 0

    # Frequencies.
    for i in range(n_vals):
        bins_dens_xy[bins_ts_x[i], bins_ts_y[i]] += 1

    # Densities.
    for i in range(n_bins):
        for j in range(n_bins):
            bins_dens_xy[i, j] /= n_vals
    return


cpdef void fill_etpy_lcl_ts(
        const DT_UL[::1] bins_ts_x,
        const DT_UL[::1] bins_ts_y,
        const DT_D[::1] bins_dens_x,
        const DT_D[::1] bins_dens_y,
              DT_D[::1] etpy_ts,
        const DT_D[:, ::1] bins_dens_xy) except +:

    cdef:
        Py_ssize_t i
        DT_D dens, prod
        DT_UL n_vals = bins_ts_x.shape[0], n_bins = bins_dens_xy.shape[0]

    assert bins_dens_x.shape[0] == n_bins
    assert bins_dens_y.shape[0] == n_bins
    assert bins_dens_xy.shape[1] == n_bins
    assert n_vals == bins_ts_y.shape[0]
    assert n_vals == etpy_ts.shape[0]

    for i in range(n_vals):    
        etpy_ts[i] = 0

    for i in range(n_vals):
        dens = bins_dens_xy[bins_ts_x[i], bins_ts_y[i]]

        if not dens:
            continue

        prod = bins_dens_x[bins_ts_x[i]] * bins_dens_y[bins_ts_y[i]]

        etpy_ts[i] = (dens * log(dens / prod))

    return


cpdef void fill_bi_var_cop_dens(
        DT_D[:] x_probs, DT_D[:] y_probs, DT_D[:, ::1] emp_dens_arr) except +:

    '''Fill the bivariate empirical copula'''

    cdef:
        Py_ssize_t i, j
        Py_ssize_t tot_pts = x_probs.shape[0]
        Py_ssize_t n_cop_bins = emp_dens_arr.shape[0]

        Py_ssize_t tot_sum, i_row, j_col

        DT_D u1, u2

    assert x_probs.size == y_probs.size

    assert emp_dens_arr.shape[0] == emp_dens_arr.shape[1]

    for i in range(n_cop_bins):
        for j in range(n_cop_bins):
            emp_dens_arr[i, j] = 0.0

    tot_sum = 0
    for i in range(tot_pts):
        u1 = x_probs[i]
        u2 = y_probs[i]

        i_row = <Py_ssize_t> (u2 * n_cop_bins)
        j_col = <Py_ssize_t> (u1 * n_cop_bins)

        emp_dens_arr[i_row, j_col] += 1
        tot_sum += 1

    assert tot_pts == tot_sum

    for i in range(n_cop_bins):
        for j in range(n_cop_bins):
            emp_dens_arr[i, j] /= float(tot_pts)

    return


cpdef void fill_cumm_dist_from_bivar_emp_dens(
        DT_D[:, ::1] emp_dens_arr, DT_D[:, ::1] cum_emp_dens_arr) except +:

    cdef:
        Py_ssize_t i, j
        DT_UL rows_cols
        DT_D cum_emp_dens

    rows_cols = emp_dens_arr.shape[0]

    for i in range(1, rows_cols + 1):
        for j in range(1, rows_cols + 1):
            cum_emp_dens = 0.0
            cum_emp_dens = cum_emp_dens_arr[i - 1, j]
            cum_emp_dens += cum_emp_dens_arr[i, j - 1]
            cum_emp_dens -= cum_emp_dens_arr[i - 1, j - 1]

            cum_emp_dens += emp_dens_arr[i - 1, j - 1]

            cum_emp_dens_arr[i, j] = cum_emp_dens

    return


cpdef tuple sample_from_2d_ecop(
        const DT_D[:, ::1] ecop_dens_arr,
        const DT_UL nvs,
        const DT_UL init_idx) except +:

    '''
    Sample from an empirical copula. Each u in the copula is sampled exactly
    once.

    Parameters
    ----------
    ecop_dens_arr : 2D array of floats
        Is the relative frequency histogram i.e. empirical copula.
    nvs : integer
        The number of total points that created the ecop_dens_arr. The
        resulting time series will have a length of nvs as well.
    init_idx : integer
        A value between 0 and nvs - 1. This is the index of the starting point.
        The first sampled u will be (init_idx + 1) / (nvs + 1).

    Returns
    -------
    us : 1d array of floats
        Time series of probabilities.
    idx_freqs : 1D array of integers
        The record of indices that were sampled already. This algorithm aims
        at sampling each point from the empirical copula only once.
    '''

    cdef:
        Py_ssize_t i, j, ctr, min_diff_idx, idx1, idx2, ui

        DT_UL n_sel_idxs
        DT_UL n_ext_vals
        DT_UL beg, end
        DT_UL all_ones_flag
        DT_UL n_zero_idxs
        DT_UL n_sample_idxs

        DT_UL n_bins = ecop_dens_arr.shape[0]
        DT_UL n_us_per_bin = nvs // n_bins 
        DT_UL max_vals_per_cond_dist = n_bins * n_us_per_bin

        DT_D diff, min_diff
        DT_D ecops_ps_ext_sum
        DT_D rand_real
        DT_D u1, u2
        DT_UL ecop_u1_idx

        DT_UL[::1] idxs_freqs = np.zeros(nvs, dtype=np.uint32)

        DT_UL[::1] sel_idxs = np.empty(n_bins, dtype=np.uint32)

        DT_UL[::1] sel_idxs_ext = np.empty(
            max_vals_per_cond_dist, dtype=np.uint32)

        DT_UL[::1] zero_idxs = np.empty(
            max_vals_per_cond_dist, dtype=np.uint32)

        DT_UL[::1] sample_idxs = np.empty(
            max_vals_per_cond_dist, dtype=np.uint32)

        DT_D[::1] ecop_ps = np.empty(n_bins, dtype=np.float64)

        DT_D[::1] ecop_ps_ext = np.empty(
            max_vals_per_cond_dist, dtype=np.float64)

        DT_D[::1] sample_ecop_ps = np.empty(
            max_vals_per_cond_dist + 1, dtype=np.float64)

        DT_D[::1] us = np.empty(nvs, dtype=np.float64)

    assert 0 <= init_idx <= nvs - 1, 'init_idx out of bounds!'

    
    # This is important, otherwise the conditional probabilities may not
    # sum up to one which might be a problem.
    assert nvs % n_bins == 0, 'nvs not a multiple of n_bins!'

    idx1 = init_idx

    u1 = (idx1 + 1.0) / (nvs + 1.0)

    us[0] = u1

    idxs_freqs[idx1] += 1
    for ui in range(1, nvs):
        ecop_u1_idx = <DT_UL> (u1 * n_bins)

        # Rescale densities tso they sum up to one.
        for i in range(n_bins):
            ecop_ps[i] = ecop_dens_arr[i, ecop_u1_idx] * n_bins

        # Choose indices of those bins where density is non-zero.
        ctr = 0
        for i in range(n_bins):
            if ecop_ps[i] == 0:
                continue

            sel_idxs[ctr] = i
            ctr += 1

        n_sel_idxs = ctr

        assert n_sel_idxs, 'No indicies selected!'

        # Accounts for multiple indices per bin. This way all indices are sampled
        # and not just the starting ones relating to each bin.
        ctr = 0
        n_ext_vals = n_sel_idxs * n_us_per_bin
        for i in range(n_sel_idxs):
            beg = <DT_UL> ((sel_idxs[i] * nvs) / <DT_D> n_bins)
            end = <DT_UL> (beg + n_us_per_bin)

            for j in range(beg, end):
                sel_idxs_ext[ctr] = j

                # Densities are uniformly redistributed for all indices that fall
                # in a given bin.
                ecop_ps_ext[ctr] = ecop_ps[sel_idxs[i]] / n_us_per_bin

                ctr += 1

        assert ctr == n_ext_vals, (
            'Did not write n_ext_vals (%d, %d)!' % (ctr, n_ext_vals))

        # Check for indices that are sampled from bins already. The idea is to
        # sample each index exactly once.
        ctr = 0
        all_ones_flag = 0
        for i in range(n_ext_vals):
            ctr += <DT_UL> idxs_freqs[sel_idxs_ext[i]]

        if ctr == n_ext_vals:
            all_ones_flag = 1

        # Find indices that are not sampled yet.
        ctr = 0
        for i in range(nvs):
            if idxs_freqs[i] == 0:
                zero_idxs[ctr] = i
                ctr += 1

        n_zero_idxs = ctr
        assert 0 < n_zero_idxs <= max_vals_per_cond_dist, (
            'Too many or too few n_zero_idxs (%d, %d)!' % (
                n_zero_idxs, max_vals_per_cond_dist))

        if all_ones_flag:
            # If all the indices in a given conditional distribution are sampled
            # from then, look for the closest one to any of them such that it
            # is the closest.

            # Nearest that is not sampled.
            min_diff = INFINITY
            for i in range(n_zero_idxs):
                for j in range(n_ext_vals):
                    diff = (<DT_D> zero_idxs[i] - sel_idxs_ext[j])**2
    
                    if diff < min_diff:
                        min_diff_idx = i
                        min_diff = diff

            idx2 = zero_idxs[min_diff_idx]

        else:
            # Based on densities in each bin, sample from only those ones that are
            # not sampled yet.
            ctr = 0
            ecops_ps_ext_sum = 0
            for i in range(n_ext_vals):
                if idxs_freqs[sel_idxs_ext[i]] == 0:
                    sample_idxs[ctr] = i
                    sample_ecop_ps[ctr] = ecop_ps_ext[i]
                    ecops_ps_ext_sum += ecop_ps_ext[i]

                    ctr += 1

            n_sample_idxs = ctr

            # Cumsum to get cdf.
            for i in range(1, n_sample_idxs):
                sample_ecop_ps[i] += sample_ecop_ps[i - 1]

            for i in range(n_sample_idxs):
                sample_ecop_ps[i] /= ecops_ps_ext_sum

            for i in range(n_sample_idxs - 1, -1, -1):
                sample_ecop_ps[i + 1] = sample_ecop_ps[i]

            sample_ecop_ps[0] = 0

            assert sample_ecop_ps[n_sample_idxs] == 1, (
                'Selected empirical densities do not integrate to 1 (%f)!' % (
                    sample_ecop_ps[n_sample_idxs]))

            rand_real = rand_c()
            for i in range(n_sample_idxs):
                if sample_ecop_ps[i] <= rand_real < sample_ecop_ps[i + 1]:
                    idx2 = sel_idxs_ext[sample_idxs[i]]
                    break

        assert idxs_freqs[idx2] == 0, (
            'Non-zero idxs_freqs[idx2] (%d)!' % idxs_freqs[idx2])

        u2 = ((idx2 + 1.0) / (nvs + 1.0))

        assert 0 < u2 < 1, '%f out of bounds (idx2: %d)!' % (u2, idx2)

        idx1 = idx2

        u1 = u2

        idxs_freqs[idx1] += 1

        us[ui] = u1

    assert np.all(np.asarray(idxs_freqs) == 1), 'idxs_freqs are not all ones!'

    return np.asarray(us), np.asarray(idxs_freqs)


cpdef DT_D get_etpy_min(DT_UL n_bins) except +:

    cdef:
        DT_D dens, etpy

#     dens = 1 / n_bins
#
#     etpy = -log(dens)

    etpy = 0.0 * n_bins

    return etpy


cpdef DT_D get_etpy_max(DT_UL n_bins):

    cdef:
        DT_D dens, etpy

    dens = (1 / (n_bins ** 2))

    etpy = -log(dens)

    return etpy


cdef void fill_nd_hist(
        const DT_D[:, ::1] probs,
        unsigned long long[::1] hist_nd,
        unsigned long long n_bins) nogil:

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
        unsigned long long n_bins):

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


cpdef np.ndarray get_nd_ecop(
        const DT_D[:, ::1] probs, 
        unsigned long long n_bins) except +:

    cdef:
        unsigned long long i, j, n_cells, n_vals, n_dims, idx, idx_temp

        unsigned long long[::1] idxs

        unsigned long long[::1] hist_nd, hist_nd_intg

        DT_D[::1] ecop

    n_vals = probs.shape[0]
    n_dims = probs.shape[1]

    assert n_vals > 1, 'n_vals too low!'
    assert n_dims > 1, 'n_dims too low!'
    assert n_bins > 1, 'n_bins too low!'
    assert chk_uexp_ovf(n_bins, n_dims) == 0, (
        'Number of cells for the ecop too large!')

    n_cells = n_bins ** n_dims

    hist_nd = np.zeros(n_cells, dtype=np.uint64)
    hist_nd_intg = np.zeros(n_cells, dtype=np.uint64)

    fill_nd_hist(probs, hist_nd, n_bins)
    fill_hist_nd_integral(hist_nd, hist_nd_intg, n_dims, n_bins)

    del hist_nd

    ecop = np.zeros(n_cells, dtype=np.float64)
    for i in range(n_cells):
        ecop[i] = <DT_D> hist_nd_intg[i] / n_vals

    return np.asarray(ecop)


cpdef get_hist_nd(
        const DT_D[:, ::1] probs, 
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

    fill_nd_hist(probs, hist_nd, n_bins)

    return np.asarray(hist_nd)
