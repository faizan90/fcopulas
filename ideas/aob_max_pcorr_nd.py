'''
@author: Faizan-Uni-Stuttgart

Oct 27, 2022

5:12:28 PM

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

from fcopulas import fill_probs_dists_v2_cy

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\Doktor_Uni_Stuttgart\04_Writing\04_Thesis\data\iaaft_sims__mmiaaft\dis_daily_1961_2015__420_3421_3470_3465_4427')

    os.chdir(main_dir)

    fig_size = (10, 7)
    dpi = 150
    alpha = 0.4
    #==========================================================================

    obs_file = Path('data/cross_sims_ref.csv')

    asymms_obs = get_asymms_max_fts(obs_file)

    asymms_sims = []

    for sim_file in main_dir.glob('./data/cross_sims_[0-9]*.csv'):
        asymms_sims.append(get_asymms_max_fts(sim_file))

    n_asymms = len(asymms_obs)

    x_crds = (asymms_obs[0].size * 2) / (
        np.arange(1, asymms_obs[0].size + 1))

    plt.figure(figsize=fig_size)

    for asymm_idx in range(n_asymms):

        print('Asymm_idx:', asymm_idx)

        leg_flag = True
        for asymms_sim in asymms_sims:

            if leg_flag:
                label = 'GAU'
                leg_flag = False

            else:
                label = None

            plt.semilogx(
                x_crds,
                asymms_sim[asymm_idx],
                alpha=alpha,
                c='b',
                label=label)

        plt.semilogx(
            x_crds, asymms_obs[asymm_idx], alpha=0.9, c='r', label='OBS')

        plt.xlabel(f'Period (steps)')
        plt.ylabel(f'Cummulative FT correlation')

        plt.xlim(plt.xlim()[::-1])

        plt.title(f'{asymm_idx}th asymmetry, nD: {n_asymms - 1}')

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.legend()

        plt.savefig(
            f'test_max_ft_asymms_cmpr_{asymm_idx}.png',
            bbox_inches='tight',
            dpi=dpi)

        plt.cla()

    plt.close()

    return


def get_asymms_max_fts(in_file):

    in_df = pd.read_csv(in_file, sep=';', index_col=0)

    assert np.all(np.isfinite(in_df.values))

    in_data = in_df.values

    in_probs = np.apply_along_axis(
        rankdata, 0, in_data) / (in_data.shape[0] + 1.0)

    in_probs = in_probs.copy(order='c')

    max_fts = []

    probs_dists = np.empty(in_probs.shape, order='c')
    for i in range(in_probs.shape[1] + 1):

        fill_probs_dists_v2_cy(in_probs, probs_dists, i)

        probs_dists **= 3

        max_fts.append(get_ms_cross_cumm_corrs_ft(probs_dists))

    return max_fts


def get_ms_cross_cumm_corrs_ft(data):

    '''
    Multivariate cross cummulative maximum correlation.
    '''

    assert data.ndim == 2

    mags_spec = np.abs(np.fft.rfft(data, axis=0)[1:,:])

    cumm_corrs = mags_spec.prod(axis=1).cumsum()

    norm_val = (
        (mags_spec ** mags_spec.shape[1]).sum(axis=0).prod()
        ) ** (1.0 / mags_spec.shape[1])

    cumm_corrs /= norm_val

    return cumm_corrs


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
