'''
@author: Faizan-Uni-Stuttgart

Nov 3, 2021

2:56:46 PM

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

from fcopulas.cyth import get_srho_minus_for_probs_nd

np.set_printoptions(
    precision=3,
    linewidth=200000,
    formatter={'float': '{:+0.3f}'.format})

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\Copulas_practice\ecop_nd')
    os.chdir(main_dir)

    in_data_file = Path(
        r'neckar_full_neckar_avg_temp_kriging_1961-01-01_to_2015-12-31_1km_all__EDK.csv')

    # in_data_file = Path(
    #     r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    denom_add = 1.0  # only applies to the python verision.
    #==========================================================================

    in_df = pd.read_csv(in_data_file, sep=';', index_col=0)

    # in_df = in_df[['2446', '420', '1462', '427']] # Plochingen.
    # in_df = in_df[['2452', '4422', '3421']] # Enz.
    # in_df = in_df[['46358', '4428', '3465']] # Kocher.
    # in_df = in_df[[ '1412', '477', '478', '3470']]  # Jagst.
    # in_df = in_df[['420', '427', '3421']]

    # in_df = in_df[[ '1412', '478']]
    in_df = in_df[[ '1412', '478', '3470']]
    # in_df = in_df[[ '1412', '478', '3470', '427']]
    # in_df = in_df[[ '1412', '478', '3470', '427', '3421']]

    assert np.all(np.isfinite(in_df.values))

    # Station cmpr.
    in_data = in_df.values

    ranks = rankdata(in_data, method='average', axis=0).astype(int)

    ranks = ranks.copy('c')

    probs = ranks / (in_data.shape[0] + denom_add)

    probs = probs.copy('c')

    n_dims = probs.shape[1]

    n_vals = probs.shape[0]

    if False:
        print('Using np.arange as probs!')
        probs = np.tile(
            np.arange(1, n_vals + 1).reshape(-1, 1),
            (1, n_dims)) / (n_vals + denom_add)

    # probs[:, 0] = 1 - probs[:, 0]
    # probs[:, 1] = 1 - probs[:, 1]
    # probs[:, 2] = 1 - probs[:, 2]
    #==========================================================================

    print('scorr mat:')
    print(np.corrcoef(probs.T))
    #==========================================================================

    # probs_le_cts = np.zeros(n_vals)
    # for i in range(n_vals):
    #     u = probs[i,:]
    #
    #     probs_le_cts[i] = (probs <= u).all(axis=1).sum()
    #
    # probs_le_cts /= n_vals

    scorr_minus_cyth = get_srho_minus_for_probs_nd(probs)

    print(scorr_minus_cyth)

    return


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack,
    # 2. "up" move the stack up to an older frame,
    # 3. "down" move the stack down to a newer frame, and
    # 4. "interact" start an interactive interpreter.
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
#
