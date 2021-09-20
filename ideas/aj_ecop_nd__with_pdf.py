'''
@author: Faizan-Uni-Stuttgart

Aug 13, 2021

9:55:21 AM

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
import matplotlib.pyplot as plt; plt.ioff()

from fcopulas import (
    get_nd_ecop,
    get_srho_for_ecop_nd,
    get_srho_plus_for_hist_nd,
    get_hist_nd)

DEBUG_FLAG = False

np.set_printoptions(
    precision=4,
    linewidth=200000,
    formatter={'float': '{:+0.4f}'.format})


def main():

    '''
    The lower orthant (rho_minus) seems to rho_plus with 1-probs as the input.
    '''

    main_dir = Path(r'P:\Synchronize\IWS\Testings\Copulas_practice\ecop_nd')
    os.chdir(main_dir)

    in_data_file = Path(
        r'neckar_full_neckar_avg_temp_kriging_1961-01-01_to_2015-12-31_1km_all__EDK.csv')

#     in_data_file = Path(
#         r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    in_df = pd.read_csv(in_data_file, sep=';', index_col=0)

#         in_df = in_df[['2446', '420', '1462', '427']] # Plochingen.
#         in_df = in_df[['2452', '4422', '3421']] # Enz.
#         in_df = in_df[['46358', '4428', '3465']] # Kocher.
#         in_df = in_df[[ '1412', '477', '478', '3470']]  # Jagst.
#         in_df = in_df[['420', '427', '3421']]

    in_df = in_df[[ '1412', '478']]
#     in_df = in_df[[ '1412', '478', '3470']]
#     in_df = in_df[[ '1412', '478', '3470', '427']]
#     in_df = in_df[[ '1412', '478', '3470', '427', '1462']]
#     in_df = in_df[[ '1412', '478', '3470', '427', '1462', '420']]

#     in_df = in_df[[ '1412', '1412']]  # , '1412']]

#     in_df = in_df.loc['1990-01-01':'2000-12-31']

#         in_df['1412'] = get_flipped_vals(in_df['1412'].values)
#         in_df['477'] = get_flipped_vals(in_df['477'].values)
#         in_df['478'] = get_flipped_vals(in_df['478'].values)
#         in_df['3470'] = get_flipped_vals(in_df['3470'].values)

    assert np.all(np.isfinite(in_df.values))

    # Station cmpr.
    in_data = in_df.values

    probs = np.apply_along_axis(
        rankdata, 0, in_data) / (in_data.shape[0] + 1.0)

#     probs = np.random.random(size=probs.shape)
#     probs[:, 1] = 1 - probs[:, 1][::-1]

    probs = probs.copy('c')

    n_dims = probs.shape[1]

    n_vals = probs.shape[0]

    n_bins = 70

    print('Using np.arange as probs!')
    probs = np.tile(
        np.arange(1, n_vals + 1).reshape(-1, 1),
        (1, n_dims)) / (n_vals + 1.0)

#     probs = 1 - probs

#     probs[:, 1] = probs[:, 1][::-1]
#     probs[:, 2] = probs[:, 2][::-1]

    print('scorr mat:')
    scorr_mat = np.corrcoef(probs.T)
    print(scorr_mat)

    #==========================================================================

    if True:
        print(n_vals, n_dims, n_bins)
#         print(probs)

    n_cells = n_bins ** n_dims
    #==========================================================================

    # Compute mean spearman correlation.
    mean_corr = []
    for i in range(n_dims):
        for j in range(n_dims):
            if j <= i:
                continue

            mean_corr.append(scorr_mat[i, j])

    print('mean_scorr:', np.array(mean_corr).mean())
    #==========================================================================

    if False:
        # Compute relative frequencies for the empirical copula.
        beg_time = timeit.default_timer()

        ecop = np.zeros(n_cells)

        for i in range(n_vals):
            idxs = np.zeros(n_dims, dtype=np.uint64)
            for j in range(n_dims):
                idxs[j] = int(probs[i, j] * n_bins)

                if j:
                    idxs[j] *= n_bins ** j

            if False:
                print(idxs, idxs.sum())

            ecop[idxs.sum(dtype=np.uint64)] += 1

        end_time = timeit.default_timer()

        print(f'Histogram took: {end_time - beg_time:0.5f} seconds.')

    #     hist_time_pyth = end_time - beg_time

        ecop /= n_vals

        ecop_reshape = ecop.reshape(*[n_bins] * n_dims)

        if False:
            print('ecop')
            print(ecop_reshape)
            print(ecop.sum())

    #==========================================================================

    if False:
        beg_time = timeit.default_timer()

        print('ecop from pdf...')
        ecop_pyth = np.zeros(n_cells)

        sclrs = np.array(
            [n_bins ** i for i in range(n_dims)], dtype=np.uint64)

        idxs = np.zeros(n_dims, dtype=np.uint64)
        for i in range(n_cells):
            us_prod = 1.0
            for j in range(n_dims):
                us_prod *= (((i // sclrs[j]) % n_bins) + 1) / (n_bins + 1.0)
                idxs[j] = ((i // sclrs[j]) % n_bins)

            ecop_pyth[i] = us_prod * ecop[i]

            print(idxs, us_prod)

        rho_plus = (n_dims + 1.0) / ((2 ** n_dims) - (n_dims + 1))
        rho_plus *= (((2 ** n_dims) * ecop_pyth.sum()) - 1)
        end_time = timeit.default_timer()
        print('rho_plus:', rho_plus, end_time - beg_time)
    #==========================================================================

    if False:
        beg_time = timeit.default_timer()
        ecop_cyth = get_nd_ecop(probs, n_bins)
        scorr_cyth = get_srho_for_ecop_nd(ecop_cyth, n_dims)
        end_time = timeit.default_timer()

        print('scorr cyth:', scorr_cyth, end_time - beg_time)

    if True:

#         beg_time = timeit.default_timer()

        ecop_hist = get_hist_nd(probs, n_bins)

        beg_time = timeit.default_timer()

        rho_plus_cyth = get_srho_plus_for_hist_nd(
            ecop_hist,
            n_dims,
            n_bins,
            n_vals)

        end_time = timeit.default_timer()

        print('rho_plus_cyth:', rho_plus_cyth, end_time - beg_time)

#     if n_dims == 3:
#         for i in range(n_bins):
#             plt.imshow(
#                 cudus_reshape[i,:,:],
#                 origin='lower',
#                 cmap='jet',
#                 vmin=0,
#                 vmax=1,
#                 alpha=0.7)
#
#             plt.title(i)
#             plt.colorbar()
#             plt.show()
#
#             plt.close()
#
#     elif n_dims == 2:
#         plt.imshow(
#             cudus_reshape,
#             origin='lower',
#             cmap='jet',
#             vmin=0,
#             vmax=1,
#             alpha=0.7)
#
#         plt.colorbar()
#         plt.show()
#
#         plt.close()

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
