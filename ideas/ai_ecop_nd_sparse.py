'''
@author: Faizan-Uni-Stuttgart

Aug 25, 2021

12:05:34 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt; plt.ioff()

from fcopulas import get_nd_ecop, get_srho_for_ecop_nd

DEBUG_FLAG = False

np.set_printoptions(
    precision=3,
    linewidth=200000,
    formatter={'float': '{:+0.3f}'.format})


def get_fh_bds(n_vals, n_dims):

    probs = np.arange(1.0, n_vals + 1) / (n_vals + 1.0)

    probs_grid = np.meshgrid(*[probs] * n_dims, copy=False)

    for i in range(n_dims):
        probs_grid[i] = probs_grid[i].ravel().reshape(-1, 1)

    probs_grid = np.concatenate(probs_grid, axis=1)

    fh_lb = np.maximum((1 - n_dims) + probs_grid.sum(axis=1), 0)
    fh_ub = probs_grid.min(axis=1)

    return fh_lb, fh_ub


def gen_symm_corr_matrix(n_dims, corr_min, corr_max):

    corrs = np.zeros((n_dims, n_dims))

    corrs.ravel()[::(n_dims + 1)] = 1
    for i in range(n_dims):
        corrs[i, i + 1:] = corr_min + (
            (corr_max - corr_min) * np.random.random(n_dims - i - 1))

        corrs[i + 1:, i] = corrs[i, i + 1:]

    # To check if the matrix is symmetric.
    if True:
        for i in range(corrs.shape[0]):
            assert np.all(corrs[i,:] == corrs[:, i]), (i,
                corrs[i,:], corrs[:, i])

    return corrs


def main():

    '''
    CDF locations at all points of observations. This may result in the cdf
    value not integrating to one as there is no C(1, 1) for negatively
    correlated variables.
    '''

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    main_dir = Path(r'P:\Synchronize\IWS\Testings\Copulas_practice\ecop_nd')
    os.chdir(main_dir)

    in_data_file = Path(
        r'neckar_full_neckar_avg_temp_kriging_1961-01-01_to_2015-12-31_1km_all__EDK.csv')

    in_df = pd.read_csv(in_data_file, sep=';', index_col=0)

    in_df = in_df[[ '1412', '478']].iloc[:365 * 3]
#     in_df = in_df[[ '1412', '478', '3470']]
#     in_df = in_df[[ '1412', '478', '3470', '427']]

    assert np.all(np.isfinite(in_df.values))

    in_data = in_df.values

    n_dims = in_data.shape[1]

    n_vals = in_data.shape[0]

    print('data shape:', in_data.shape)

#     fh_lb, fh_ub = get_fh_bds(n_vals, n_dims)

    # For reps.
    # in_data = np.round(in_data, 0)

    # For no reps.
#     in_data += np.random.random(size=(n_vals, n_dims)) * 1e-3

    # Random.
#     in_data = np.random.random(size=(n_vals, n_dims))

    # Fully correlated data.
    in_data = np.tile(
        np.arange(1, n_vals + 1).reshape(-1, 1),
        (1, n_dims)) / (n_vals + 1.0)

    # Gaussian.
    #==========================================================================
#     print('Generating corr that is semi-positive definite...')
#
#     min_corr = -(1 - 2e-1)
#     max_corr = -(1 - 2e-1)
#
#     ctr = 0
#     while True:
#         ctr += 1
#
#         corrs = gen_symm_corr_matrix(n_dims, min_corr, max_corr)
#         eig_vals = np.linalg.eigvals(corrs)
#
#         if not np.all(eig_vals >= 0):
#             continue
#
#         else:
#             break
#
#     print('corr mat:')
#     print(corrs)
#
#     in_data = np.random.multivariate_normal(
#         np.zeros(n_dims), corrs, size=n_vals)
    #==========================================================================

    ranks = np.apply_along_axis(rankdata, 0, in_data)

    idxs = ranks - 1

#     probs = ranks / (n_vals + 1.0)
#     probs = probs.copy('c')

    idxs_df = pd.DataFrame(
        columns=np.arange(n_dims), data=idxs, dtype=np.int64)

    # Sorting like this is important for the second version to work.
    idxs_df.sort_values(
        by=np.arange(n_dims).tolist(), axis='index', inplace=True)

    idx_vals = idxs_df.values

#==============================================================================
    # Look back at all the values. The most inefficient version.
#     cdf_vals = np.zeros(n_vals)
#     beg_time = timeit.default_timer()
#
#     ctr_a = 0
#     for i in range(n_vals):
#         cidx = idx_vals[i]
#
#         for j in range(n_vals):
#             pidx = idx_vals[j]
#
#             ctr_a += 1
#
#             if not np.all(pidx <= cidx):
#                 continue
#
#             cdf_vals[i] += 1 / n_vals
#
#     end_time = timeit.default_timer()
#
#     print('ctr_a:', ctr_a)
#     print(f'Python time: {end_time - beg_time:0.3f} secs')
#
# #     print(cdf_vals)
#     print(cdf_vals.min(), cdf_vals.max())
#==============================================================================

#==============================================================================
    # Look back at some values. Slightly efficient version.
    # This is two times faster than the raw version.
    cdf_vals = np.zeros(n_vals)
    beg_time = timeit.default_timer()

    ctr_b = 0
    for i in range(n_vals):
        cidx = idx_vals[i]

        for j in range(i + 1):
            pidx = idx_vals[j]
            ctr_b += 1
            if not np.all(pidx <= cidx):
                continue

            cdf_vals[i] += 1 / n_vals

    end_time = timeit.default_timer()

    print('ctr_b:', ctr_b)
    print(f'Python time: {end_time - beg_time:0.3f} secs')
#     print(cdf_vals)
    print(cdf_vals.min(), cdf_vals.max())
#==============================================================================

#     probs = (idxs_df.values + 1.0) / (n_vals + 1.0)
#
#     if n_dims == 2:
#
#         fig = plt.figure()
#
#         ax = fig.add_subplot(111, projection='3d')
#
#         ax.scatter(
#             probs[:, 0], probs[:, 1], cdf_vals, alpha=0.99, c=cdf_vals)
#
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('CDF')
#
#         plt.show()

    return

#==============================================================================
#     cdf_vals = np.full(n_vals, 1 / n_vals).cumsum()

#     ranks_dists = ((np.zeros((1, n_dims)) - probs) ** 2).prod(axis=1)
#
#     rank_dist_idxs = np.argsort(ranks_dists)
#
#     probs_sort = probs[rank_dist_idxs,:]

#     if n_dims == 2:
#
#         fh_lb = np.maximum(probs_sort.sum(axis=1) - 1, 0)
#         fh_ub = probs_sort.min(axis=1)
#
#         print(np.any(cdf_vals < fh_lb))
#         print(np.any(cdf_vals > fh_ub))
#         print(np.any(fh_lb > fh_ub))
#
#         fig = plt.figure()
#
#         ax = fig.add_subplot(111, projection='3d')
#
# #         ax.scatter(
# #             probs_sort[:, 0], probs_sort[:, 1], fh_lb, alpha=0.7, c='k')
#
#         ax.scatter(
#             probs_sort[:, 0], probs_sort[:, 1], fh_ub, alpha=0.99, c='b')
#
#         ax.scatter(
#             probs_sort[:, 0], probs_sort[:, 1], cdf_vals, alpha=0.99, c=cdf_vals)
#
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('CDF')
#
#         plt.show()
#==============================================================================


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack
    # 2. "up" move the stack up to an older frame
    # 3. "down" move the stack down to a newer frame
    # 4. "interact" start an interactive interpreter
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

            import pdb
            pdb.post_mortem()
    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
