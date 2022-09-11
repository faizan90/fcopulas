'''
@author: Faizan-Uni-Stuttgart

Aug 18, 2022

6:56:24 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt; plt.ioff()

from kde import KERNEL_FTNS_DICT

DEBUG_FLAG = False


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    half_window_size = 100
    #==========================================================================

    rel_dists_sym = np.concatenate((
        np.arange(half_window_size, -1, -1.0,),
        np.arange(1.0, half_window_size + 1.0)))

    rel_dists_sym = np.concatenate((rel_dists_sym, rel_dists_sym))

    rel_dists_sym /= half_window_size + 1.0

    # rel_dists_asym = np.concatenate((
    #     np.linspace(half_window_size, 0, half_window_size // 4),
    #     np.arange(1.0, (half_window_size * 2) - (half_window_size // 4) + 2.0)))

    # rel_dists_asym = np.concatenate((
    #     np.repeat(half_window_size, half_window_size),
    #     np.arange(1.0, (half_window_size * 2) - (half_window_size // 4) + 2.0)))

    rel_dists_asym = np.concatenate((
        np.repeat(half_window_size, half_window_size // 4),
        np.linspace(0, 150, 100)))

    rel_dists_asym /= half_window_size + 1.0
    # rel_dists_asym /= rel_dists_asym.max() + 1.0
    # rel_dists_asym /= 10

    # assert rel_dists_asym.size == rel_dists_sym.size

    # kern_ftn = KERNEL_FTNS_DICT['epan']
    # kern_ftn = KERNEL_FTNS_DICT['sigm']
    # kern_ftn = KERNEL_FTNS_DICT['logi']
    # kern_ftn = KERNEL_FTNS_DICT['tric']
    kern_ftn = KERNEL_FTNS_DICT['triw']
    # kern_ftn = KERNEL_FTNS_DICT['biwe']

    ref_ser_sym = kern_ftn(rel_dists_sym)
    ref_ser_sym /= ref_ser_sym.max()

    ref_ser_asym = kern_ftn(rel_dists_asym)
    ref_ser_asym /= ref_ser_asym.max()

    convo = np.convolve(ref_ser_asym, ref_ser_sym, mode='same')
    # convo = np.convolve(ref_ser_asym, ref_ser_asym[::-1], mode='full')

    convo /= convo.max()

    plt.plot(ref_ser_sym, alpha=0.7, lw=2, c='r', label='sym')
    plt.plot(ref_ser_asym, alpha=0.7, lw=2, c='b', label='asym')
    plt.plot(convo, alpha=0.7, lw=2, c='g', label='convo')

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.legend()

    plt.xlabel('Time step')
    plt.ylabel('Magnitude')

    plt.show()

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
