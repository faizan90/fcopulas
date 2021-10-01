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

DEBUG_FLAG = False


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)
    #==========================================================================

#     vtype = 2

    #==========================================================================

    for vtype in (1, 2):

        if vtype == 1:
            # Precipitation data.
            in_file = Path(
                r'P:\Synchronize\IWS\Testings\Copulas_practice\ecop_diagonal_densities\rockenau_ppt_1961_2015_lumped.csv')

            sep = ';'

            # labels = '420;427;3465;3470;3421;454'.split(';')
#             labels = '420;3465;3421'.split(';')
            labels = '3465;3421'.split(';')

            beg_time = '1961-01-01'
            end_time = '2015-12-31'

            n_round = 1
        #==========================================================================

        elif vtype == 2:
            # Discharge data.
            in_file = Path(
                r'P:\Synchronize\IWS\Testings\Copulas_practice\ecop_diagonal_densities\neckar_daily_discharge_1961_2015.csv')

            sep = ';'

            # labels = '420;427;3465;3470;3421;454'.split(';')
#             labels = '420;3465;3421'.split(';')
            labels = '3465;3421'.split(';')

            beg_time = '1961-01-01'
            end_time = '2015-12-31'

            n_round = 1
        #==========================================================================

        else:
            raise NotImplementedError(f'Unknown vtype: {vtype}')

        data_df = pd.read_csv(in_file, sep=sep, index_col=0)

        data_df = data_df.loc[beg_time:end_time, labels]

        assert np.isfinite(data_df.values).all()

        data_df = data_df.round(n_round)

        print(data_df.shape)

        probs_nd = (
            data_df.rank(axis=0) / (data_df.shape[0] + 1.0)
            ).values.copy(order='c')

        print(probs_nd.shape)

        n_vals, n_dims = probs_nd.shape

        diagonal_denss = np.zeros(n_vals)

        for i in range(n_vals):

            diagonal_denss[i] = (
                probs_nd <= ((i + 1.0) / (n_vals + 1.0))).all(axis=1).sum()

        print(diagonal_denss)

        diagonal_denss /= (n_vals + 1.0)

        plt.plot(
            np.linspace(1 / (n_vals + 1.0), n_vals / (n_vals + 1.0), n_vals),
            diagonal_denss,
            alpha=0.75,
            label=f'vtype: {vtype}')

    plt.plot([0, 1], [0, 1], alpha=0.75, label='ideal')

    plt.legend()

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.gca().set_aspect('equal')

    plt.xlabel('Threshold density')
    plt.ylabel('Diagonal CDF value')

    plt.title(f'nD: {n_dims}')

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
