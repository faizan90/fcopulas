'''
@author: Faizan-Uni-Stuttgart

Aug 12, 2021

3:51:04 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
from scipy.stats import norm

DEBUG_FLAG = False

np.set_printoptions(
    precision=3,
    linewidth=200000,
    formatter={'float': '{:+0.3f}'.format})


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

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    n_dims = 4
    sample_size = int(1e6)

    min_corr = 1 - 1e-1
    max_corr = 1 - 1e-7

    thresh_probs = [
        0, 0.5, 0.6, 0.7, 0.8, 0.9,
        1 - 1e-2, 1 - 1e-3, 1 - 1e-4, 1 - 1e-5, 1 - 1e-6, 1 - 1e-7]

    min_thresh_vals = 50
    #==========================================================================

    print('Generating corr that is semi-positive definite...')
    ctr = 0
    while True:
        ctr += 1

        corrs = gen_symm_corr_matrix(n_dims, min_corr, max_corr)
        eig_vals = np.linalg.eigvals(corrs)

        if not np.all(eig_vals >= 0):
            continue

        else:
            break

    print(f'Needed {ctr} tries to succeed!')
    print('corrs:')
    print(corrs)

    assert corrs.shape[0] == n_dims
    assert corrs.shape[1] == n_dims
    print('#' * 80)
    #==========================================================================

    print('Generating gau_vecs...')
    means = np.zeros(n_dims)

    gau_vecs = np.random.multivariate_normal(means, corrs, size=sample_size)

    print('#' * 80)

    print('Generating probs...')
    probs = norm.cdf(gau_vecs)

    print('#' * 80)
    #==========================================================================

    print('corrs for tail thresholds:')
    print('#' * 80)
    print('')

    for thresh_prob in thresh_probs:
        print('thresh_prob:', thresh_prob)

        take_idxs = np.zeros(sample_size, dtype=bool)
        for i in range(n_dims):
            take_idxs |= probs[:, i] >= thresh_prob

        # On very large sample size, overflow occurs and the sum is wrong.
        # So use the correct datatype.
        n_take_idxs = take_idxs.sum(dtype=np.uint64)

        print('n_take_idxs:', n_take_idxs)
        print('%age n_take_idxs:', round(100 * n_take_idxs / sample_size, 3))

        if n_take_idxs < min_thresh_vals:
            print('Not enough values to work with!')
            print('#' * 80)
            print('')
            continue

        tail_corrs = np.corrcoef(gau_vecs[take_idxs,:].T)

        print('tail_corrs:')
        print(tail_corrs)

        print('tail_corrs - corrs:')
        print(tail_corrs - corrs)

        print('#' * 80)
        print('')

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
