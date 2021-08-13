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

DEBUG_FLAG = False

np.set_printoptions(
    precision=3,
    linewidth=200000,
    formatter={'float': '{:+0.3f}'.format})


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    main_dir = Path(r'P:\Synchronize\IWS\Testings\Copulas_practice\ecop_nd')
    os.chdir(main_dir)

    in_data_file = Path(
        r'neckar_full_neckar_avg_temp_kriging_1961-01-01_to_2015-12-31_1km_all__EDK.csv')

    in_df = pd.read_csv(in_data_file, sep=';', index_col=0)

#         in_df = in_df[['2446', '420', '1462', '427']] # Plochingen.
#         in_df = in_df[['2452', '4422', '3421']] # Enz.
#         in_df = in_df[['46358', '4428', '3465']] # Kocher.
#         in_df = in_df[[ '1412', '477', '478', '3470']]  # Jagst.
#         in_df = in_df[['420', '427', '3421']]
    in_df = in_df[[ '1412', '478', '3470']]

    in_df = in_df.loc['1990-01-01':'2000-12-31']

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
#     probs[:, 1] = probs[:, 1][::-1]

    n_dims = probs.shape[1]

    n_vals = probs.shape[0]

    n_bins = 12
    #==========================================================================

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

        if False:
            print(idxs, idxs.sum())

        ecop[idxs.sum(dtype=np.uint64)] += 1

    ecop /= n_vals

    ecop_reshape = ecop.reshape(*[n_bins] * n_dims)

    if False:
        print(ecop_reshape)
        print(ecop.sum())
    #==========================================================================

    print('\n\nCumm')
    # Cummulative sum of ecop array.
    sclr_arr = np.array(
        [n_bins ** i for i in range(0, n_dims)], dtype=np.uint64)

    cudus = np.zeros_like(ecop)

    beg_time = timeit.default_timer()
    for i in range(n_cells):
        idxs = np.zeros(n_dims, dtype=np.uint64)

        # Get indices back for a given index in ecop.
        for j in range(n_dims - 1, -1, -1):
            idxs[j] = int(i / (n_bins ** j)) % n_bins

        if False:
            print(i, idxs)  # * sclr_arr

        cudu = 0.0
        cidxs = np.zeros(n_dims, dtype=np.uint64)

        ctr = 0
        while ctr <= i:
            for j in range(n_dims - 1, -1, -1):
                cidxs[j] = int(ctr / (n_bins ** j)) % n_bins

                # Small opt. went from 7.6 to 5.6 secs.
                if cidxs[j] > idxs[j]:
#                     ctr += 1
                    break

#             cudu += ecop[(cidxs * sclr_arr).sum()]

            if np.all(cidxs <= idxs):
                cudu += ecop[(cidxs * sclr_arr).sum()]

            ctr += 1

        cudus[i] = cudu

    end_time = timeit.default_timer()

    print(f'Cumm took: {end_time - beg_time:0.2f} seconds.')

    cudus_reshape = cudus.reshape(*[n_bins] * n_dims)

    print(cudus_reshape)

#     for i in range(n_bins):
#         plt.imshow(
#             cudus_reshape[i,:,:],
# #             ecop_reshape[i,:,:],
#             origin='lower',
#             cmap='jet',
#             vmin=0,
#             vmax=1,
#             alpha=0.7)
#
#         plt.title(i)
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
