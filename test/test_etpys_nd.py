'''
@author: Faizan-Uni-Stuttgart

Sep 20, 2021

6:04:16 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np

from fcopulas import (
    get_hist_nd, get_etpy_nd, get_etpy_max_nd, get_etpy_min_nd)

DEBUG_FLAG = False


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    n = int(1e7)

    n_dims = 4

    n_bins = 20
    #==========================================================================

    assert n_bins ** n_dims < n

    etpy_nd_min = get_etpy_min_nd(n_bins, n_dims)
    etpy_nd_max = get_etpy_max_nd(n_bins, n_dims)

    print('Minimum possible entropy:', etpy_nd_min)
    print('Maximum possible entropy:', etpy_nd_max)

    # Fully correlated case.
    probs = np.linspace(1 / (n + 1), n / (n + 1), n)

    probs = np.concatenate([probs.reshape(-1, 1)] * n_dims, axis=1)

    hist_nd = get_hist_nd(probs, n_bins)

    etpy_nd = get_etpy_nd(hist_nd, n)

    print('Entropy of fully correlated data:', etpy_nd)

    # Random data.
    probs = np.random.random(size=(n, n_dims))

    hist_nd = get_hist_nd(probs, n_bins)

    etpy_nd = get_etpy_nd(hist_nd, n)

    print('Entropy of random data:', etpy_nd)
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
