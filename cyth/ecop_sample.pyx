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

cdef extern from "./rand_gen_mp.h" nogil:
    cdef:
        DT_D rand_c()
        void warm_up()  # Call this everytime.

warm_up()


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

        DT_UL[::1] idxs_freqs = np.zeros(nvs, dtype=np.uint64)

        DT_UL[::1] sel_idxs = np.empty(n_bins, dtype=np.uint64)

        DT_UL[::1] sel_idxs_ext = np.empty(
            max_vals_per_cond_dist, dtype=np.uint64)

        DT_UL[::1] zero_idxs = np.empty(
            max_vals_per_cond_dist, dtype=np.uint64)

        DT_UL[::1] sample_idxs = np.empty(
            max_vals_per_cond_dist, dtype=np.uint64)

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

            # Fully random selection of next index.
    #         idx2 = np.random.choice(zero_idxs)  # TODO.

            # Nearest that is zero.
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


# cpdef tuple get_cond_idx2_2d_ecop(
#         const DT_D[:, ::1] ecop_dens_arr,
#         const DT_UL[::1] idxs_freqs,
#         const DT_D u1) except +:
# 
#     '''
#     Sample the index for the next step from a given empirical copula
#     density array and the first variable. The conditions is that each index
#     i.e. next u2 has to be sampled only once. Read further comments for more.
# 
#     Parameters
#     ----------
#     ecop_dens_arr : 2D array of floats
#         Is the relative frequency histogram i.e. empirical copula.
#     u1 : float
#         The value to use as the predictor to draw a conditonal distribution
#         from the emirical copula.
#     idx_freqs : 1D array of integers
#         The record of indices that were sampled already. This algorithm aims
#         at sampling each point from the empirical copula only once.
# 
#     Returns
#     -------
#     idx2 : integer
#         The index of u2 that follows the given u1.
#         Can be converted to u2 later.
#     u2 : float
#         The next u in the copula.
#     '''
# 
#     cdef:
#         Py_ssize_t i, j, ctr, min_diff_idx, idx2
# 
#         DT_UL n_sel_idxs
#         DT_UL n_ext_vals
#         DT_UL beg, end
#         DT_UL all_ones_flag
#         DT_UL n_zero_idxs
#         DT_UL n_sample_idxs
# 
#         DT_UL n_bins = ecop_dens_arr.shape[0]
#         DT_UL nvs = idxs_freqs.shape[0]
#         DT_UL ecop_u1_idx = <DT_UL> (u1 * n_bins)
#         DT_UL n_us_per_bin = nvs // n_bins 
#         DT_UL max_vals_per_cond_dist = n_bins * n_us_per_bin
# 
#         DT_D diff, min_diff
#         DT_D ecops_ps_ext_sum
#         DT_D rand_real
#         DT_D u2
# 
#         DT_UL[::1] sel_idxs = np.empty(n_bins, dtype=np.uint64)
# 
#         DT_UL[::1] sel_idxs_ext = np.empty(
#             max_vals_per_cond_dist, dtype=np.uint64)
# 
#         DT_UL[::1] zero_idxs = np.empty(
#             max_vals_per_cond_dist, dtype=np.uint64)
# 
#         DT_UL[::1] sample_idxs = np.empty(
#             max_vals_per_cond_dist, dtype=np.uint64)
# 
#         DT_D[::1] ecop_ps = np.empty(n_bins, dtype=np.float64)
# 
#         DT_D[::1] ecop_ps_ext = np.empty(
#             max_vals_per_cond_dist, dtype=np.float64)
# 
#         DT_D[::1] sample_ecop_ps = np.empty(
#             max_vals_per_cond_dist + 1, dtype=np.float64)
# 
#     # This is important, otherwise the conditional probabilities may not
#     # sum up to one which might be a problem.
#     assert nvs % n_bins == 0, 'nvs not a multiple of n_bins!'
# 
#     # Rescale densities tso they sum up to one.
#     for i in range(n_bins):
#         ecop_ps[i] = ecop_dens_arr[i, ecop_u1_idx] * n_bins
# 
#     # Choose indices of those bins where density is non-zero.
#     ctr = 0
#     for i in range(n_bins):
#         if ecop_ps[i] == 0:
#             continue
# 
#         sel_idxs[ctr] = i
#         ctr += 1
# 
#     n_sel_idxs = ctr
# 
#     assert n_sel_idxs, 'No indicies selected!'
# 
#     # Accounts for multiple indices per bin. This way all indices are sampled
#     # and not just the starting ones relating to each bin.
#     ctr = 0
#     n_ext_vals = n_sel_idxs * n_us_per_bin
#     for i in range(n_sel_idxs):
#         beg = <DT_UL> ((sel_idxs[i] * nvs) / <DT_D> n_bins)
#         end = <DT_UL> (beg + n_us_per_bin)
# 
#         for j in range(beg, end):
#             sel_idxs_ext[ctr] = j
# 
#             # Densities are uniformly redistributed for all indices that fall
#             # in a given bin.
#             ecop_ps_ext[ctr] = ecop_ps[sel_idxs[i]] / n_us_per_bin
# 
#             ctr += 1
# 
#     assert ctr == n_ext_vals, (
#         'Did not write n_ext_vals (%d, %d)!' % (ctr, n_ext_vals))
#     
#     # Check for indices that are sampled from bins already. The idea is to
#     # sample each index exactly once.
#     ctr = 0
#     all_ones_flag = 0
#     for i in range(n_ext_vals):
#         ctr += <DT_UL> idxs_freqs[sel_idxs_ext[i]]
# 
#     if ctr == n_ext_vals:
#         all_ones_flag = 1
# 
#     # Find indices that are not sampled yet.
#     ctr = 0
#     for i in range(nvs):
#         if idxs_freqs[i] == 0:
#             zero_idxs[ctr] = i
#             ctr += 1
# 
#     n_zero_idxs = ctr
#     assert 0 < n_zero_idxs <= max_vals_per_cond_dist, (
#         'Too many or too few n_zero_idxs (%d, %d)!' % (
#             n_zero_idxs, max_vals_per_cond_dist))
# 
#     if all_ones_flag:
#         # If all the indices in a given conditional distribution are sampled
#         # from then, look for the closest one to any of them such that it
#         # is the closest.
# 
#         # Fully random selection of next index.
# #         idx2 = np.random.choice(zero_idxs)  # TODO.
# 
#         # Nearest that is zero.
#         min_diff = INFINITY
#         for i in range(n_zero_idxs):
#             for j in range(n_ext_vals):
#                 diff = (<DT_D> zero_idxs[i] - sel_idxs_ext[j])**2
# 
#                 if diff < min_diff:
#                     min_diff_idx = i
#                     min_diff = diff
# 
#         idx2 = zero_idxs[min_diff_idx]
# 
#     else:
#         # Based on densities in each bin, sample from only those ones that are
#         # not sampled yet.
#         ctr = 0
#         ecops_ps_ext_sum = 0
#         for i in range(n_ext_vals):
#             if idxs_freqs[sel_idxs_ext[i]] == 0:
#                 sample_idxs[ctr] = i
#                 sample_ecop_ps[ctr] = ecop_ps_ext[i]
#                 ecops_ps_ext_sum += ecop_ps_ext[i]
# 
#                 ctr += 1
# 
#         n_sample_idxs = ctr
# 
#         # Cumsum to get cdf.
#         for i in range(1, n_sample_idxs):
#             sample_ecop_ps[i] += sample_ecop_ps[i - 1]
# 
#         for i in range(n_sample_idxs):
#             sample_ecop_ps[i] /= ecops_ps_ext_sum
# 
#         for i in range(n_sample_idxs - 1, -1, -1):
#             sample_ecop_ps[i + 1] = sample_ecop_ps[i]
# 
#         sample_ecop_ps[0] = 0
# 
#         assert sample_ecop_ps[n_sample_idxs] == 1, (
#             'Selected empirical densities do not integrate to 1 (%f)!' % (
#                 sample_ecop_ps[n_sample_idxs]))
# 
#         rand_real = rand_c()
#         for i in range(n_sample_idxs):
#             if sample_ecop_ps[i] <= rand_real < sample_ecop_ps[i + 1]:
#                 idx2 = sel_idxs_ext[sample_idxs[i]]
#                 break
# 
#     assert idxs_freqs[idx2] == 0, (
#         'Non-zero idxs_freqs[idx2] (%d)!' % idxs_freqs[idx2])
# 
#     u2 = ((idx2 + 1.0) / (nvs + 1.0))
# 
#     assert 0 < u2 < 1, '%f out of bounds (idx2: %d)!' % (u2, idx2)
# 
#     return idx2, u2
