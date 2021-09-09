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

from fcopulas import get_nd_ecop, get_srho_for_ecop_nd

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

#     in_df = in_df[[ '1412', '478']]
    in_df = in_df[[ '1412', '478', '3470']]
#     in_df = in_df[[ '1412', '478', '3470', '427']]

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

    n_bins = 30

#     print('scorr mat:')
#     print(np.corrcoef(probs.T))

    print('Using np.arange as probs!')
    probs = np.tile(
        np.arange(1, n_vals + 1).reshape(-1, 1),
        (1, n_dims)) / (n_vals + 1.0)

    #==========================================================================

    if True:
        print(n_vals, n_dims, n_bins)
        print(probs)
    #==========================================================================

#     # Compute relative frequencies for the empirical copula.
#     beg_time = timeit.default_timer()
#
#     n_cells = n_bins ** n_dims
#     ecop = np.zeros(n_cells)
#
#     for i in range(n_vals):
#         idxs = np.zeros(n_dims, dtype=np.uint64)
#         for j in range(n_dims):
#             idxs[j] = int(probs[i, j] * n_bins)
#
#             if j:
#                 idxs[j] *= n_bins ** j
#
#         if False:
#             print(idxs, idxs.sum())
#
#         ecop[idxs.sum(dtype=np.uint64)] += 1
#
#     end_time = timeit.default_timer()
#
#     print(f'Histogram took: {end_time - beg_time:0.5f} seconds.')
#
# #     hist_time_pyth = end_time - beg_time
#
# #     ecop /= n_vals
#
#     ecop_reshape = ecop.reshape(*[n_bins] * n_dims)
#
#     if True:
#         print('ecop')
#         print(ecop_reshape)
#         print(ecop.sum())

    #==========================================================================

#     return

#     print('\n\nCumm')
#     # Cummulative sum of ecop array.
#     sclr_arr = np.array(
#         [n_bins ** i for i in range(0, n_dims)], dtype=np.uint64)
#
#     cudus = np.zeros_like(ecop)
#
#     beg_time = timeit.default_timer()
#     for i in range(n_cells):
#         idxs = np.zeros(n_dims, dtype=np.uint64)
#
#         # Get indices back for a given index in ecop.
#         for j in range(n_dims):
#             idxs[j] = int(i / (n_bins ** j)) % n_bins
#
#         if False:
#             print(i, idxs)  # * sclr_arr
#
#         cudu = 0.0
#         cidxs = np.zeros(n_dims, dtype=np.uint64)
#
#         ctr = 0
#         while ctr <= i:
#             # This can be further optimized.
#             sel_flag = True
#             for j in range(n_dims):
#                 cidxs[j] = int(ctr / (n_bins ** j)) % n_bins
#
#                 # Small opt. went from 7.6 to 5.6 secs.
#                 if cidxs[j] > idxs[j]:
#                     sel_flag = False
#                     break
#
# #             if np.all(cidxs <= idxs):
#             if sel_flag:
#                 cudu += ecop[(cidxs * sclr_arr).sum()]
#
#             ctr += 1
#
#         cudus[i] = cudu
#
#     end_time = timeit.default_timer()
#
# #     print(cudus)
#
#     cudus_reshape = cudus.reshape(*[n_bins] * n_dims)
#
#     print(cudus_reshape)
#
#     print(f'Cumm took: {end_time - beg_time:0.2f} seconds.')
    #==========================================================================

#     print('\n\nCumm_fast')
#     # Cummulative sum of ecop array.
#     sclr_arr = np.array(
#         [n_bins ** i for i in range(0, n_dims)], dtype=np.uint64)
#
#     cudus_fast = np.zeros_like(ecop)
# #     cudus_fast[0] = ecop[0]
#
#     beg_time = timeit.default_timer()
#     for i in range(n_cells):
#         idxs = np.zeros(n_dims, dtype=np.uint64)
#         bidxs = np.zeros(n_dims, dtype=np.uint64)
#
#         # Get indices back for a given index in ecop.
#         for j in range(n_dims):
#             idxs[j] = int(i / (n_bins ** j)) % n_bins
#
#             if idxs[j]:
#                 bidxs[j] = idxs[j] - 1
#
#         cudu = 0.0
#
#         # TIP: Cummulative count of cells helps. The number of cells
#         # to each cell must be the product of the indices + 1.
#
#         #======================================================================
#
#         # First approach.
# #         cudu = ecop[(idxs * sclr_arr).sum()]
# #         bd_ctr = 0
# #         for j in range(n_dims):
# #             cidxs = idxs.copy()
# #
# #             if idxs[j]:
# #                 cidxs[j] -= 1
# #
# #             else:
# #                 continue
# #
# # #             if not cidxs[j]:
# # #                 continue
# #
# #             bd_ctr += 1
# #             cudu += cudus_fast[(cidxs * sclr_arr).sum()]
# #
# #         # Forward rotation.
# #         for j in range(n_dims - 1):
# #             cidxs = idxs.copy()
# #
# #             if not (idxs[j] and idxs[j + 1]):
# #                 continue
# #
# #             if idxs[j]:
# #                 cidxs[j] -= 1
# #
# #             if idxs[j + 1]:
# #                 cidxs[j + 1] -= 1
# #
# #             cudu -= cudus_fast[(cidxs * sclr_arr).sum()]
# #
# #             if False:
# #                 print(i, idxs, cudu, cudus_fast[(cidxs * sclr_arr).sum()])
# #
# #         # Remaining rotation.
# #         if idxs[0] and idxs[n_dims - 1]:
# #             cidxs = idxs.copy()
# #
# #             j = 0
# #             if idxs[j]:
# #                 cidxs[j] -= 1
# #
# #             j = n_dims - 1
# #             if idxs[j]:
# #                 cidxs[j] -= 1
# #
# #             cudu -= cudus_fast[(cidxs * sclr_arr).sum()]
#
# #         if bd_ctr > 1:
# #             cudu -= cudus_fast[(bidxs * sclr_arr).sum()] * (bd_ctr - 1)
#         #======================================================================
#
#         # Second approach.
# #         cudu = ecop[(idxs * sclr_arr).sum()]
# #
# # #         if np.all(idxs):
# #         cudu += cudus_fast[(bidxs * sclr_arr).sum()]
# #
# #         for j in range(n_dims):
# #             cidxs = idxs.copy()
# #             for k in range(idxs[j]):
# #                 cidxs[j] = k
# #                 cudu += ecop[(cidxs * sclr_arr).sum()]
#
#         # Some more steps needed.
#         #======================================================================
#
#         # Third approach.
# #         cudu = ecop[(idxs * sclr_arr).sum()]
# #         for j in range(n_dims):
# #             cidxs = idxs.copy()
# #             for k in range(idxs[j]):
# #                 cidxs[j] = k
# #                 cudu += ecop[(cidxs * sclr_arr).sum()]
# #
# #         bd_ctr = 0
# #         # Forward rotation.
# #         for j in range(n_dims - 1):
# #             cidxs = idxs.copy()
# #
# # #             if not (idxs[j] and idxs[j + 1]):
# # #                 continue
# #
# # #             if not np.all(idxs[j:j + n_dims - 1]):
# # #                 continue
# #
# # #             cidxs[j:j + n_dims - 1] -= 1
# #
# #             t_ctr = 0
# #             for k in range(j, j + n_dims - 1):
# #                 if not idxs[k]:
# #                     continue
# #
# #                 t_ctr += 1
# #                 cidxs[k] -= 1
# #
# #             if t_ctr < n_dims - 1:
# #                 continue
# #
# # #             print(cidxs)
# #             bd_ctr += 1
# #             cudu += cudus_fast[(cidxs * sclr_arr).sum()]
# #
# #             if False:
# #                 print(i, idxs, cudu, cudus_fast[(cidxs * sclr_arr).sum()])
# #
# #         # Remaining rotation.
# #         idxs_roll = np.roll(idxs, n_dims - 2).astype(np.uint64)
# #         for j in range(n_dims - 2):
# #             cidxs = idxs_roll.copy()
# #
# # #             if not (idxs[j] and idxs[j + 1]):
# # #                 continue
# #
# # #             if not np.all(idxs_roll[j:j + n_dims - 1]):
# # #                 continue
# #
# # #             cidxs[j:j + n_dims - 1] -= 1
# #
# #             t_ctr = 0
# #             for k in range(j, j + n_dims - 1):
# #                 if not idxs_roll[k]:
# #                     continue
# #
# #                 t_ctr += 1
# #                 cidxs[k] -= 1
# #
# #             if t_ctr < n_dims - 1:
# #                 continue
# #
# #             bd_ctr += 1
# #             cidxs = np.roll(cidxs, 2 - n_dims)
# #             cudu += cudus_fast[(cidxs * sclr_arr).sum()]
# #
# #         if bd_ctr > 1:
# #             cudu -= cudus_fast[(bidxs * sclr_arr).sum()] * (bd_ctr - 1)
#
#         #======================================================================
#
#         # Fourth approach.
#         # A bit inefficient but should be faster than the basic approach.
#
#         cudu = ecop[(idxs * sclr_arr).sum()]
#
# #         for j in range(n_dims):
# #             cidxs = np.zeros_like(idxs)
# #             cidxs[j] = idxs[j]
# #
# #             cudu += ecop[(cidxs * sclr_arr).sum()]
# #
# #             k = 0
# #             while k < n_dims:
# # #                 if k == j:
# # #                     k += 1
# # #                     continue
# #
# #                 cidxs[k] += 1
# #
# #                 print(cidxs)
# #
# # #                 if cidxs[k] >= idxs[k]:
# # #                     for l in range(k + 1):
# # #                         if l == j:
# # #                             continue
# # #
# # #                         cidxs[l] = 0
# # #
# # #                     if (k + 1) == j and j < (n_dims + 1):
# # #                         k += 1
# # #                         cidxs[k + 1] += 1
# # #
# # #                     else:
# # #                         break
# #                 if cidxs[k] == n_dims:
# #                     for l in range(k + 1):
# #                         cidxs[l] = 0
# #
# #                 k += 1
#
#         #======================================================================
#
#         cudus_fast[i] = cudu
#
#         if False:
#             print(i, idxs, cudus_fast[i])
#
# #     def inc_idxs(j, cidxs, n_bins):
# #
# #         cidxs[j] += 1
# #
# #         if cidxs[j] == n_bins:
# #             j += 1
# #
# #             if j == cidxs.shape[0]:
# #                 return
# #
# #             cidxs[:j] = 0
# #             inc_idxs(j, cidxs, n_bins)
# # #             cidxs[j] += 1
# #
# #         print(cidxs)
# #         inc_idxs(j, cidxs, n_bins)
# #
# #         return
#
#     cidxs = np.zeros_like(idxs)
#     print(cidxs)
#     j = 0
# #     inc_idxs(j, cidxs, n_bins)
# #     while True:
# #         cidxs[j] += 1
# #         if j == n_dims and cidxs[j] == n_bins:
# #             break
# #
# #         if cidxs[j] == n_bins:
# #             j += 1
# #             cidxs[:j] = 0
# #             cidxs[j] += 1
# #
# #             if cidxs[j] == n_bins:
# #                 break
# #
# #             j = 0
# #
# #         print(cidxs)
#
#     end_time = timeit.default_timer()
#
# #     print(cudus_fast)
#
#     cudus_fast_reshape = cudus_fast.reshape(*[n_bins] * n_dims)
#
#     print(cudus_fast_reshape)
#
#     print(f'Cumm_fast took: {end_time - beg_time:0.2f} seconds.')
#
#     assert np.all(np.isclose(cudus, cudus_fast))
    #==========================================================================

#     beg_time = timeit.default_timer()
#
    ecop_cyth = get_nd_ecop(probs, n_bins)
#
#     end_time = timeit.default_timer()
#
#     print(ecop_cyth)
#     print(f'Ecop_cyth took: {end_time - beg_time:0.5f} seconds.')
#
#     assert np.all(np.isclose(ecop_cyth, cudus))
#
#     scorr_pyth = get_srho_for_ecop_nd(cudus, n_dims)
#     scorr_pyth_fast = get_srho_for_ecop_nd(cudus_fast, n_dims)
    scorr_cyth = get_srho_for_ecop_nd(ecop_cyth, n_dims)
#
#     print('scorr pyth:', scorr_pyth)
#     print('scorr pyth_fast:', scorr_pyth_fast)
    print('scorr cyth:', scorr_cyth)

#     cudus_reshape = cudus.reshape(*[n_bins] * n_dims)
#
#     print(cudus_reshape)

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
