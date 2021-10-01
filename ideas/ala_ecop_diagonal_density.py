'''
@author: Faizan-Uni-Stuttgart

Sep 21, 2021

3:14:54 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.ioff()

from fcopulas import (
    get_hist_nd,
    )

DEBUG_FLAG = False


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)
    #==========================================================================

    vtype = 1

    #==========================================================================

    if vtype == 1:
        # Precipitation data.
        in_file = Path(
            r'P:\Synchronize\IWS\Testings\Copulas_practice\ecop_diagonal_densities\rockenau_ppt_1961_2015_lumped.csv')

        sep = ';'

        # labels = '420;427;3465;3470;3421;454'.split(';')
        labels = '420;3465;3421'.split(';')

        beg_time = '1961-01-01'
        end_time = '2015-12-31'

        n_round = 1

        n_ecop_bins = 500

#         ecop_uthresh = None
        ecop_uthresh = 0.9
    #==========================================================================

    elif vtype == 2:
        # Discharge data.
        in_file = Path(
            r'P:\Synchronize\IWS\Testings\Copulas_practice\ecop_diagonal_densities\neckar_daily_discharge_1961_2015.csv')

        sep = ';'

        # labels = '420;427;3465;3470;3421;454'.split(';')
        labels = '420;3465;3421'.split(';')

        beg_time = '1961-01-01'
        end_time = '2015-12-31'

        n_round = 1

        n_ecop_bins = 500

#         ecop_uthresh = None
        ecop_uthresh = 0.9
    #==========================================================================

    else:
        raise NotImplementedError(f'Unknown vtype: {vtype}')

    if ecop_uthresh is not None:
        assert 0 < ecop_uthresh < 1.0

    else:
        ecop_uthresh = 0

    data_df = pd.read_csv(in_file, sep=sep, index_col=0)

    data_df = data_df.loc[beg_time:end_time, labels]

    assert np.isfinite(data_df.values).all()

    data_df = data_df.round(n_round)

    print(data_df.shape)

    probs_nd = (
        data_df.rank(axis=0) / (data_df.shape[0] + 1.0)
        ).values.copy(order='c')

#     if ecop_uthresh is not None:
#         prob_thresh_idxs = (probs_nd >= ecop_uthresh).all(axis=1)
#
#         data_df = data_df.loc[prob_thresh_idxs,:]
#
#         probs_nd = (
#             data_df.rank(axis=0) / (data_df.shape[0] + 1.0)
#             ).values.copy(order='c')

    print(probs_nd.shape)

    n_vals, n_dims = probs_nd.shape

#     assert (n_ecop_bins ** n_dims) <= n_vals, n_ecop_bins ** n_dims

    hist_nd = get_hist_nd(probs_nd, n_ecop_bins)

    diagonal_denss = np.zeros(n_ecop_bins)

    for i in range(int(ecop_uthresh * n_ecop_bins), n_ecop_bins):
        idxs = np.zeros(n_dims, dtype=np.uint64)
        for j in range(n_dims):
            idxs[j] = i

            if j:
                idxs[j] *= n_ecop_bins ** j

        diagonal_denss[i] = hist_nd[idxs.sum(dtype=np.uint64)] / float(n_vals)

    print(diagonal_denss)

    plt.plot(diagonal_denss, alpha=0.75, c='b')

    plt.xlim(int(ecop_uthresh * n_ecop_bins), n_ecop_bins)

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.xlabel('Copula diagonal bin index')
    plt.ylabel('Bin density')

    plt.title(f'No. of bins: {n_ecop_bins}')

    plt.show()
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
