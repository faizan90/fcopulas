'''
@author: Faizan-Uni-Stuttgart

Mar 30, 2021

11:37:32 AM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from phsann.cyth import get_asymms_sample, fill_bi_var_cop_dens
from phsann.misc import roll_real_2arrs

plt.ioff()

DEBUG_FLAG = True


def _get_asymm_1_max(scorr):

    a_max = (
        0.5 * (1 - scorr)) * (1 - ((0.5 * (1 - scorr)) ** (1.0 / 3.0)))

    return a_max


def _get_asymm_2_max(scorr):

    a_max = (
        0.5 * (1 + scorr)) * (1 - ((0.5 * (1 + scorr)) ** (1.0 / 3.0)))

    return a_max


def _get_etpy_min(n_bins):

    dens = 1 / n_bins

    etpy = -np.log(dens)

    return etpy


def _get_etpy_max(n_bins):

    dens = (1 / (n_bins ** 2))

    etpy = -np.log(dens)

    return etpy


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\Copulas_practice\random_error_effects')
    os.chdir(main_dir)

    in_file = Path(r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    lag_steps = np.arange(1, 31, dtype=np.int64)

    ecop_bins = 20

    fig_size = (15, 8)
    plt_alpha = 0.5

    beg_time = '1961-01-01'
    end_time = '2015-12-31'

    stn = '420'

    clrs = ['r', 'b', 'k', 'g']
    labs = ['ref', '5%', '10%', '20%']
    errs = [0.0, 0.05, 0.1, 0.2]

    err_type = 'marginals'

    out_fig_name = f'ae_cmpr__{err_type}__{stn}__{beg_time}_{end_time}.png'

    in_ser = pd.read_csv(in_file, sep=';', index_col=0)[stn]

    in_ser.index = pd.to_datetime(in_ser.index, format='%Y-%m-%d')

    in_ser = in_ser.loc[beg_time: end_time]

    etpy_min = _get_etpy_min(ecop_bins)
    etpy_max = _get_etpy_max(ecop_bins)

    ecop_dens_arrs = np.full((ecop_bins, ecop_bins), np.nan, dtype=np.float64)

    axes = plt.subplots(2, 3, squeeze=False, figsize=fig_size)[1]

    for clr, lab, err in zip(clrs, labs, errs):
        data = in_ser.values.copy()

        if err_type == 'marginals':
            data += data * err * (-1 + (2 * np.random.random(size=data.size)))

        elif err_type == 'marginals_flipped':
            idxs = rankdata(data).astype(int)

            idxs = idxs.size - idxs

            data += np.sort(data)[idxs] * (
                err * (-1 + (2 * np.random.random(size=data.size))))

        elif err_type in ('probs', 'probs_flipped', 'probs_random'):
            pass

        else:
            raise NotImplementedError(f'Unknown errtype: {err_type}!')

        probs = rankdata(data) / (data.size + 1.0)

        p_flag = False
        if err_type == 'probs':
            probs += probs * err * (
                -1 + (2 * np.random.random(size=data.size)))

            p_flag = True

        elif err_type == 'probs_flipped':
            idxs = rankdata(probs).astype(int)

            idxs = idxs.size - idxs

            probs += np.sort(probs)[idxs] * (
                err * (-1 + (2 * np.random.random(size=data.size))))

            p_flag = True

        elif err_type == 'probs_random':
            probs += err * (-1 + (2 * np.random.random(size=data.size)))

            p_flag = True

        elif err_type in ('marginals', 'marginals_flipped'):
            pass

        else:
            raise NotImplementedError(f'Unknown errtype: {err_type}!')

        probs[probs < (1 / (probs.size + 1.0))] = (1 / (probs.size + 1.0))

        probs[probs > (probs.size / (probs.size + 1.0))] = (
            probs.size / (probs.size + 1.0))

        if p_flag:
            idxs = (rankdata(probs) - 1).astype(int)
            data = np.sort(data)[idxs]

        scorrs = []
        asymms_1 = []
        asymms_2 = []
        etpys = []
        pcorrs = []
        for lag_step in lag_steps:
            probs_i, rolled_probs_i = roll_real_2arrs(probs, probs, lag_step)
            data_i, rolled_data_i = roll_real_2arrs(data, data, lag_step)

            # scorr.
            scorr = np.corrcoef(probs_i, rolled_probs_i)[0, 1]
            scorrs.append(scorr)

            # asymms.
            asymm_1, asymm_2 = get_asymms_sample(probs_i, rolled_probs_i)

            asymm_1 /= _get_asymm_1_max(scorr)

            asymm_2 /= _get_asymm_2_max(scorr)

            asymms_1.append(asymm_1)
            asymms_2.append(asymm_2)

            # ecop etpy.
            fill_bi_var_cop_dens(probs_i, rolled_probs_i, ecop_dens_arrs)

            non_zero_idxs = ecop_dens_arrs > 0

            dens = ecop_dens_arrs[non_zero_idxs]

            etpy_arr = -(dens * np.log(dens))

            etpy = etpy_arr.sum()

            etpy = (etpy - etpy_min) / (etpy_max - etpy_min)

            etpys.append(etpy)

            # pcorr.
            pcorr = np.corrcoef(data_i, rolled_data_i)[0, 1]
            pcorrs.append(pcorr)

        # plot
        axes[0, 0].plot(
            lag_steps, scorrs, alpha=plt_alpha, color=clr, label=lab)

        axes[1, 0].plot(
            lag_steps, asymms_1, alpha=plt_alpha, color=clr, label=lab)

        axes[1, 1].plot(
            lag_steps, asymms_2, alpha=plt_alpha, color=clr, label=lab)

        axes[0, 1].plot(
            lag_steps, etpys, alpha=plt_alpha, color=clr, label=lab)

        axes[0, 2].plot(
            lag_steps, pcorrs, alpha=plt_alpha, color=clr, label=lab)

    axes[0, 0].grid()
    axes[1, 0].grid()
    axes[1, 1].grid()
    axes[0, 1].grid()
    axes[0, 2].grid()
    axes[1, 2].set_axis_off()

    axes[0, 0].legend()
    axes[1, 0].legend()
    axes[1, 1].legend()
    axes[0, 1].legend()
    axes[0, 2].legend()
#     axes[1, 2].legend()

    axes[0, 0].set_ylabel('Spearman correlation')

    axes[1, 0].set_xlabel('Lag steps')
    axes[1, 0].set_ylabel('Asymmetry (Type - 1)')

    axes[1, 1].set_xlabel('Lag steps')
    axes[1, 1].set_ylabel('Asymmetry (Type - 2)')

    axes[0, 1].set_ylabel('Entropy')

    axes[0, 2].set_xlabel('Lag steps')
    axes[0, 2].set_ylabel('Pearson correlation')

    plt.tight_layout()

    plt.savefig(out_fig_name, bbox_inches='tight')
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
