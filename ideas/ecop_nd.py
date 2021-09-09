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
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = False

np.set_printoptions(
    precision=3,
    linewidth=200000,
    formatter={'float': '{:+0.3f}'.format})


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    n_dims = 2

    n_vals = 1000

    n_bins = 6

    shift_idx = 1  # Shift for roll.
    #==========================================================================

    probs = np.tile(
        np.arange(1, n_vals + 1).reshape(-1, 1),
        (1, n_dims)) / (n_vals + 1.0)

    for i in range(1, n_dims):
        probs[:, i] = np.roll(probs[:, i], i * shift_idx)

#     noise_ratio = 0.1 * (n_bins / n_vals)
#     probs += (-noise_ratio + ((2 * noise_ratio) * np.random.random(size=probs.shape)))

    if True:
        print(probs)
    #==========================================================================

    # Compute relative frequencies for the empirical copula.
    n_cells = n_bins ** n_dims
    ecop = np.zeros(n_cells)

    for i in range(n_vals):
        idxs = np.zeros(n_dims, dtype=np.uint64)
        for j in range(n_dims):
            idxs[j] = int(probs[i, j] * n_bins)

            if j:
                idxs[j] *= n_bins ** j

        if True:
            print(idxs, idxs.sum())

        ecop[idxs.sum(dtype=np.uint64)] += 1

    ecop /= n_vals

    if False:
        print(ecop.reshape(*[n_bins] * n_dims))
        print(ecop.sum())
    #==========================================================================

    print('\n\nCumm')
    # Cummulative sum of ecop array.
    sclr_arr = np.array(
        [n_bins ** i for i in range(0, n_dims)], dtype=np.uint64)

    cudus = np.zeros_like(ecop)

    print(sclr_arr)
    for i in range(n_cells):
        idxs = np.zeros(n_dims, dtype=np.uint64)

        # Get indices back for a given index in ecop.
        for j in range(n_dims - 1, -1, -1):
            idxs[j] = int(i / (n_bins ** j)) % n_bins

        print(i, idxs)  # * sclr_arr

        cudu = 0.0
        cidxs = np.zeros(n_dims, dtype=np.uint64)

        ctr = 0
        while ctr <= i:
            for j in range(n_dims - 1, -1, -1):
                cidxs[j] = int(ctr / (n_bins ** j)) % n_bins

            if np.all(cidxs <= idxs):
                cudu += ecop[(cidxs * sclr_arr).sum()]

            ctr += 1

        cudus[i] = cudu

    cudus_reshape = cudus.reshape(*[n_bins] * n_dims)

    print(cudus_reshape)

    if n_dims == 3:
        for i in range(n_bins):
            plt.imshow(
                cudus_reshape[i,:,:],
                origin='lower',
                cmap='jet',
                vmin=0,
                vmax=1,
                alpha=0.7)

            plt.title(i)
            plt.colorbar()
            plt.show()

            plt.close()

    elif n_dims == 2:
        plt.imshow(
            cudus_reshape,
            origin='lower',
            cmap='jet',
            vmin=0,
            vmax=1,
            alpha=0.7)

        plt.colorbar()
        plt.show()

        plt.close()

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
