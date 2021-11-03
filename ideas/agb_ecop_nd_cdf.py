'''
@author: Faizan-Uni-Stuttgart

Oct 1, 2021

4:17:46 PM

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

from fcopulas.cyth import get_ecop_nd

DEBUG_FLAG = False

np.set_printoptions(
    precision=3,
    linewidth=200000,
    formatter={'float': '{:+0.3f}'.format})


def shift_by_one(shift_idxs, max_idxs):

    n_dims = max_idxs.shape[0]

    for i in range(n_dims):

        if shift_idxs[i] > max_idxs[i]:
            continue

        shift_idxs[i] += 1

        idx = i

        while True:
            if shift_idxs[idx] > max_idxs[idx]:
                for j in range(idx + 1):

                    shift_idxs[j] = 0

                idx += 1

                if idx == n_dims:
                    break

                shift_idxs[idx] += 1

            else:
                break

        break

    return


def shift_by_one_ign(shift_idxs, max_idxs, ignore_idx):

    n_dims = max_idxs.shape[0]

    chg_flag = False
    for i in range(n_dims):

        if shift_idxs[i] > max_idxs[i]:
            continue

        if i == ignore_idx:
            continue

        chg_flag = True

        shift_idxs[i] += 1

        idx = i

        while True:
            if shift_idxs[idx] > max_idxs[idx]:
                for j in range(idx + 1):

                    if j == ignore_idx:
                        continue

                    shift_idxs[j] = 0

                idx += 1

                if idx == ignore_idx:
                    idx += 1

                if idx == n_dims:
                    break

                shift_idxs[idx] += 1

            else:
                break

        break

    return chg_flag


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\Copulas_practice\ecop_nd')
    os.chdir(main_dir)

    # shift_idxs = np.array([0, 1, 1])
    # max_idxs = np.array([1, 1, 1])
    # shift_by_one_ign(shift_idxs, max_idxs, 0)
    #
    # print(shift_idxs)
    #
    # return

    in_data_file = Path(
        r'neckar_full_neckar_avg_temp_kriging_1961-01-01_to_2015-12-31_1km_all__EDK.csv')

    in_df = pd.read_csv(in_data_file, sep=';', index_col=0)

#         in_df = in_df[['2446', '420', '1462', '427']] # Plochingen.
#         in_df = in_df[['2452', '4422', '3421']] # Enz.
#         in_df = in_df[['46358', '4428', '3465']] # Kocher.
#         in_df = in_df[[ '1412', '477', '478', '3470']]  # Jagst.
#         in_df = in_df[['420', '427', '3421']]

    # in_df = in_df[[ '1412', '478']]
    # in_df = in_df[[ '1412', '478', '3470']]
    # in_df = in_df[[ '1412', '478', '3470', '427']]
    # in_df = in_df[[ '1412', '478', '3470', '427', '1462']]

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

    n_bins = 10

    # print('scorr mat:')
    # print(np.corrcoef(probs.T))

    # print('Using np.arange as probs!')
    # probs = np.tile(
    #     np.arange(1, n_vals + 1).reshape(-1, 1),
    #     (1, n_dims)) / (n_vals + 1.0)

    #==========================================================================

    if False:
        print(n_vals, n_dims, n_bins)
        print(probs)
    #==========================================================================

#     # Compute relative frequencies for the empirical copula.
#     beg_time = timeit.default_timer()
#
#     n_cells = n_bins ** n_dims
#     ecop = np.zeros(n_cells, dtype=np.uint64)
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
#     print(f'Histogram took: {end_time - beg_time:0.6f} seconds.')
#
# #     hist_time_pyth = end_time - beg_time
#
# #     ecop /= n_vals
#
#     ecop_reshape = ecop.reshape(*[n_bins] * n_dims)
#
#     if False:
#         print('ecop')
#         print(ecop_reshape)
#         print(ecop.sum())
#     #==========================================================================
#
#     print('\n\nCumm')
#     # Cummulative sum of ecop array.
#     sclr_arr = np.array(
#         [n_bins ** i for i in range(0, n_dims)], dtype=np.int64)
#
#     cudus = np.zeros_like(ecop)
#
#     zeros = np.zeros(n_dims, dtype=np.int64)
#     beg_time = timeit.default_timer()
#
#     rpt_ctr = 0
#     tot_ctr = 0
#     excpt_ctr = 0
#     for i in range(n_cells):
#         idxs = np.zeros(n_dims, dtype=np.int64)
#
#         # Get indices back for a given index in ecop.
#         for j in range(n_dims):
#             idxs[j] = int(i // (n_bins ** j)) % n_bins
#
#         if False:
#             print(i, idxs)  # * sclr_arr
#
#         if False:
#             cudu = 0.0
#             cidxs = np.zeros(n_dims, dtype=np.int64)
#
#             ctr = 0
#             while ctr <= i:
#                 # This can be further optimized.
#                 sel_flag = True
#                 for j in range(n_dims):
#                     cidxs[j] = int(ctr / (n_bins ** j)) % n_bins
#
#                     # Small opt. went from 7.6 to 5.6 secs.
#                     if cidxs[j] > idxs[j]:
#                         sel_flag = False
#                         break
#
#                 if sel_flag:
#                     cudu += ecop[(cidxs * sclr_arr).sum()]
#
#                 ctr += 1
#
#             cudus[i] = cudu
#         #======================================================================
#
#         # Right now, the idea is to take the cdf value from cell
#         # that is directly behind and take the ecop hist values
#         # for the remaining ones. Cells that have been visited must
#         # be accounted for and no cell should be visited twice.
#         # Corner cases are taken care of.
#         else:
#             # Was four times faster when using 20 bins in 3D.
#             # Should become faster and faster as the number of bins
#             # increase, compared to the primitive version.
#
#             hind_idxs = np.maximum(zeros, idxs - 1)
#             cudu = cudus[(hind_idxs * sclr_arr).sum(dtype=np.uint64)]
#
#             # if i == (np.array([1, 1, 1, 0]) * sclr_arr).sum():
#             #     _ = 1
#
#                 # print(ecop_reshape)
#                 # print(cudus.reshape(*[n_bins] * n_dims))
#
#             visited_cells = np.zeros(i, dtype=int)
#
#             # Keep track that one thing is not visited more than once per
#             # iteration.
#             for k in range(n_dims):
#                 if not idxs[k]:
#                     continue
#
#                 max_idxs = idxs.copy()
#
#                 if (k + 1) == n_dims:
#                     max_idxs[0] = max(idxs[0] - 1, 0)
#                     # max_idxs[0] = idxs[0] - 1
#
#                 else:
#                     max_idxs[k + 1] = max(0, idxs[k + 1] - 1)
#                     # max_idxs[k + 1] = idxs[k + 1] - 1
#
#                 if np.any(max_idxs < 0):
#                     continue
#
#                 shift_idxs = np.zeros(n_dims, dtype=np.int64)
#
#                 shift_idxs[k] = idxs[k]
#
#                 # assert not np.all(shift_idxs == hind_idxs), (i, k, shift_idxs, hind_idxs)
#                 # assert not np.all(shift_idxs == idxs), (i, k, shift_idxs, idxs)
#
#                 while (max_idxs - shift_idxs).sum() > 0:
#
#                     ecop_idx = (shift_idxs * sclr_arr).sum(dtype=np.uint64)
#
#                     if not visited_cells[ecop_idx]:
#                         cudu += ecop[ecop_idx]
#                         visited_cells[ecop_idx] += 1
#
#                     else:
#                         rpt_ctr += 1
#
#                     tot_ctr += 1
#
#                     if shift_by_one_ign(shift_idxs, max_idxs, k):
#                         pass
#
#                     else:
#                         break
#
#                 ecop_idx = (shift_idxs * sclr_arr).sum(dtype=np.uint64)
#
#                 if ecop_idx != i:
#                     if not visited_cells[ecop_idx]:
#                         cudu += ecop[ecop_idx]
#                         visited_cells[ecop_idx] += 1
#
#                     else:
#                         rpt_ctr += 1
#
#                     tot_ctr += 1
#
#                 else:
#                     excpt_ctr += 1
#
#             cudus[i] = cudu + ecop[(idxs * sclr_arr).sum(dtype=np.uint64)]
#
#     end_time = timeit.default_timer()
#
#     cudus_reshape = cudus.reshape(*[n_bins] * n_dims)
#
#     print(tot_ctr, rpt_ctr, excpt_ctr)
#
#     # print(cudus_reshape)
#
#     print(f'Cumm took: {end_time - beg_time:0.6f} seconds.')
    #==========================================================================

    beg_time = timeit.default_timer()

    cudus_cyth = get_ecop_nd(probs, n_bins)

    end_time = timeit.default_timer()

    # cudus_cyth_reshape = cudus_cyth.reshape(*[n_bins] * n_dims)

    # print(cudus_cyth_reshape)

    print(f'Cython took: {end_time - beg_time:0.6f} seconds.')

    # print(cudus_reshape == cudus_cyth_reshape)

    # assert (cudus_reshape == cudus_cyth_reshape).all()

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
