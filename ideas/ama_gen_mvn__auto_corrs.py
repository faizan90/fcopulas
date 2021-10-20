'''
@author: Faizan-Uni-Stuttgart

Oct 5, 2021

4:08:21 PM

'''
import os

os.environ[str('MKL_NUM_THREADS')] = str(1)
os.environ[str('NUMEXPR_NUM_THREADS')] = str(1)
os.environ[str('OMP_NUM_THREADS')] = str(1)

import sys
import time
import timeit
import traceback as tb
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

DEBUG_FLAG = False


def perturb_corr_matrix(n_dims, ref_corrs, perturb_lim):

    corrs = np.zeros((n_dims, n_dims))

    corrs.ravel()[::(n_dims + 1)] = 1
    for i in range(n_dims):
        corrs[i, i + 1:] = ref_corrs[i, i + 1:] + perturb_lim * (
            -1.0 + (2.0 * np.random.random(n_dims - i - 1)))

        corrs[i, i + 1:] = np.maximum(-1, corrs[i, i + 1:])
        corrs[i, i + 1:] = np.minimum(+1, corrs[i, i + 1:])

        corrs[i + 1:, i] = corrs[i, i + 1:]

    # To check if the matrix is symmetric.
    if True:
        for i in range(corrs.shape[0]):
            assert np.all(corrs[i,:] == corrs[:, i]), (i,
                corrs[i,:], corrs[:, i])

    return corrs


def get_reshuffled_mvn_data(data, means, corrs, n_dims, sorted_data):

    mvn_data = np.random.multivariate_normal(
        means, corrs, size=data.shape[0])

    reshuffle_idxs = np.argsort(np.argsort(mvn_data, axis=0), axis=0)

    reshuffled_data = np.empty_like(data)

    for i in range(n_dims):
        reshuffled_data[:, i] = sorted_data[reshuffle_idxs[:, i], i]

    return reshuffled_data


def reshuffle_based_on_mvn_v2(data, perturb_lim, min_err):

    '''
    Find the mvn correlation matrix such that the reshuffled data has
    the same correlation matrix as the reference.

    The idea is that we want the same correlation for a process that
    is Gaussian and the reference (may not be Gaussian necessarily).
    '''

    n_dims = data.shape[1]

    means = np.zeros(n_dims)
    ref_corrs = np.corrcoef(data.T)

    eig_vals = np.linalg.eigvals(ref_corrs)

    assert np.all(eig_vals >= 0), eig_vals

    sorted_data = np.sort(data, axis=0)

    data_sel = get_reshuffled_mvn_data(
        data, means, ref_corrs, n_dims, sorted_data)

    # Once more. It gives a better first approximation. Helps make it faster.
    data_sel = get_reshuffled_mvn_data(
        data_sel, means, ref_corrs, n_dims, sorted_data)

    shuff_corrs = np.corrcoef(data_sel.T)

    sim_corrs_sel = shuff_corrs  # ref_corrs.copy()

    min_sq_sum_diff = ((ref_corrs - shuff_corrs) ** 2).sum()

    min_sq_sum_diff_glbl = min_sq_sum_diff

    print('Initial min_sq_sum_diff:', min_sq_sum_diff)

    ctr = 0
    sing_mat_ctr = 0
    while min_sq_sum_diff > min_err:
        ctr += 1

        sim_corrs = perturb_corr_matrix(
            n_dims,
            sim_corrs_sel,
            perturb_lim * (min_sq_sum_diff / min_sq_sum_diff_glbl) ** 0.5,
            )

        eig_vals = np.linalg.eigvals(sim_corrs)

        if not np.all(eig_vals >= 0):
#             print(sing_mat_ctr)
            sing_mat_ctr += 1
            continue

        reshuffled_data = get_reshuffled_mvn_data(
            data_sel, means, sim_corrs, n_dims, sorted_data)

        shuff_corrs = np.corrcoef(reshuffled_data.T)

        sq_diff_sum = ((ref_corrs - shuff_corrs) ** 2).sum()

#         print(ctr, sq_diff_sum)

        if sq_diff_sum < min_sq_sum_diff:
            min_sq_sum_diff = sq_diff_sum

            sim_corrs_sel = sim_corrs

            print(ctr, 'updated min_sq_sum_diff:', min_sq_sum_diff)

            data_sel = reshuffled_data

    print(ctr, 'min_sq_sum_diff:', min_sq_sum_diff)
    print('sing_mat_ctr:', sing_mat_ctr)

#     print('ref_corrs:\n', ref_corrs)
#     print('sim_corrs_sel:\n', sim_corrs_sel)
    return data_sel


def lag_var(in_arr, fin_lag):

    assert isinstance(in_arr, np.ndarray)
    assert in_arr.ndim == 2

    assert isinstance(fin_lag, int)
    assert fin_lag >= 0

    assert in_arr.shape[1]
    assert in_arr.shape[0] > fin_lag

    if not fin_lag:
        out_arr = in_arr.copy()

    else:
        out_arr = []

        for i in range(fin_lag):
            out_arr.append(in_arr[i:-(fin_lag - i)])

        out_arr = np.concatenate(out_arr, axis=1)

    return out_arr


def simulate_mvn(args):

    in_data, perturb_lim, min_err, out_dir, sim_idx, overwrite_flag = args

    out_file = out_dir / f'sim_{sim_idx}.csv'

    if (not overwrite_flag) and out_file.exists():
        return

    mvn_data = reshuffle_based_on_mvn_v2(in_data, perturb_lim, min_err)

    pd.DataFrame(data=mvn_data).to_csv(out_file, sep=';')
    return


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\Copulas_practice\asymms_nd')
    os.chdir(main_dir)

    in_data_file = Path(
        r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    read_col = '420'

    n_lags = 5

    n_sims = 1000

    n_cpus = 7

    # The more different something is than the Gaussian, the more the
    # perturb_lim.
    perturb_lim = 0.99 / n_lags
    min_err = 1e-4 * n_lags

    overwrite_flag = False

    out_dir = Path(f'mvn_discharge_auto_{read_col}_{n_lags}_{n_sims}sims2')
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    in_df = pd.read_csv(in_data_file, sep=';', index_col=0)[read_col]

    in_data = in_df.values

    in_data = lag_var(in_data.reshape(-1, 1), n_lags)

    out_file = out_dir / f'obs.csv'

    pd.DataFrame(data=in_data).to_csv(out_file, sep=';')

    mp_args = (
        (in_data, perturb_lim, min_err, out_dir, i, overwrite_flag)
        for i in range(n_sims))

    if n_cpus == 1:
        for args in mp_args:
            simulate_mvn(args)

    else:
        mp_pool = Pool(n_cpus)

        list(mp_pool.imap_unordered(simulate_mvn, mp_args, chunksize=1))

    return


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
