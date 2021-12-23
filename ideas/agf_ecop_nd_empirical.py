'''
@author: Faizan-Uni-Stuttgart

Nov 4, 2021

11:44:40 AM

'''
import os

os.environ[str('MKL_NUM_THREADS')] = str(1)
os.environ[str('NUMEXPR_NUM_THREADS')] = str(1)
os.environ[str('OMP_NUM_THREADS')] = str(1)

import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from af_mvn_tail_dependence import gen_symm_corr_matrix
from fcopulas.cyth import get_ecop_nd_empirical, get_ecop_nd_empirical_sorted

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\Copulas_practice\ecop_nd')
    os.chdir(main_dir)

    # # in_data_file = Path(
    # #     r'neckar_full_neckar_avg_temp_kriging_1961-01-01_to_2015-12-31_1km_all__EDK.csv')
    #
    # in_data_file = Path(
    #     r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')
    # #==========================================================================
    #
    # in_df = pd.read_csv(in_data_file, sep=';', index_col=0)
    #
    # # in_df = in_df[['2446', '420', '1462', '427']] # Plochingen.
    # # in_df = in_df[['2452', '4422', '3421']] # Enz.
    # # in_df = in_df[['46358', '4428', '3465']] # Kocher.
    # # in_df = in_df[[ '1412', '477', '478', '3470']]  # Jagst.
    # # in_df = in_df[['420', '427', '3421']]
    #
    # # in_df = in_df[[ '1412', '478']]
    # in_df = in_df[[ '1412', '478', '3470']]
    # # in_df = in_df[[ '1412', '478', '3470', '427']]
    # # in_df = in_df[[ '1412', '478', '3470', '427', '3421', '420']]
    #
    # assert np.all(np.isfinite(in_df.values))
    #==========================================================================

    if True:
        n_vals = int(1e6)  # in_data.shape[0]
        n_dims = 3

        min_corr = 1 - 1e-1
        max_corr = 1 - 1e-5

        print('Generating corr that is semi-positive definite...')
        ctr = 0
        while True:
            ctr += 1

            corrs = gen_symm_corr_matrix(n_dims, min_corr, max_corr)
            eig_vals = np.linalg.eigvals(corrs)

            if not np.all(eig_vals >= 0):
                continue

            else:
                break

        print(f'Needed {ctr} tries to succeed!')
        print('corrs:')
        print(corrs)

        assert corrs.shape[0] == n_dims
        assert corrs.shape[1] == n_dims
        print('#' * 80)

        print('Generating gau_vecs...')

        means = np.zeros(n_dims)

        in_df = pd.DataFrame(
            data=np.random.multivariate_normal(means, corrs, size=n_vals))

    in_data = in_df.values

    beg_time = timeit.default_timer()

    ranks = rankdata(in_data, method='max', axis=0).astype(int)
    ranks = ranks.astype(np.uint64).copy('c')

    ecdf_cyth = get_ecop_nd_empirical(ranks)

    end_time = timeit.default_timer()

    print(f'ecdf (unsorted) took {end_time - beg_time:0.6f} seconds')
    #==========================================================================

    beg_time = timeit.default_timer()

    in_df_srtd = in_df.sort_values(by=in_df.columns.tolist())

    ranks_srtd = in_df_srtd.rank(method='max').values
    ranks_srtd = ranks_srtd.astype(np.uint64).copy('c')

    ecdf_cyth_srtd = get_ecop_nd_empirical_sorted(ranks_srtd)

    ecdf_cyth_srtd_ser = pd.Series(
        index=in_df_srtd.index, data=ecdf_cyth_srtd)

    ecdf_cyth_srtd_ser = ecdf_cyth_srtd_ser.loc[in_df.index]

    end_time = timeit.default_timer()

    print(f'ecdf (sorted) took {end_time - beg_time:0.6f} seconds')
    #==========================================================================

    assert np.all(ecdf_cyth_srtd_ser == ecdf_cyth)

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
