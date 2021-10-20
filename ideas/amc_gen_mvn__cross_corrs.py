'''
@author: Faizan-Uni-Stuttgart

Oct 5, 2021

4:08:21 PM

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
from multiprocessing import Pool

import pandas as pd

from ama_gen_mvn__auto_corrs import simulate_mvn

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\Copulas_practice\asymms_nd')
    os.chdir(main_dir)

    in_data_file = Path(
        r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    read_cols = ['420', '427', '3421', '3465', '3470']

    n_sims = 1000

    n_cpus = 7

    n_read_cols = len(read_cols)

    # The more different something is than the Gaussian, the more the
    # perturb_lim.
    perturb_lim = 0.99 / n_read_cols
    min_err = 1e-4 * n_read_cols

    overwrite_flag = False

    read_cols_str = '_'.join(read_cols)

    out_dir = Path(
        f'mvn_discharge_cross_{len(read_cols)}_{read_cols_str}_{n_sims}sims')
    #==========================================================================

    assert len(read_cols) > 1

    out_dir.mkdir(exist_ok=True)

    in_df = pd.read_csv(in_data_file, sep=';', index_col=0)[read_cols]

    in_data = in_df.values

    out_file = out_dir / f'obs.csv'

    pd.DataFrame(data=in_data).to_csv(out_file, sep=';')

    mp_args = (
        (in_data, perturb_lim, min_err, out_dir, i, overwrite_flag)
        for i in range(n_sims))

    if n_cpus == 1:
        for args in mp_args:
            simulate_mvn(args)

    else:
        mp_pool = Pool(n_cpus)

        list(mp_pool.imap_unordered(simulate_mvn, mp_args, chunksize=1))

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
