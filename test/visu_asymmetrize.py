'''
@author: Faizan-Uni-Stuttgart

Aug 18, 2022

12:58:22 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, expon
import matplotlib.pyplot as plt; plt.ioff()

from kde import KERNEL_FTNS_DICT

import pyximport

pyximport.install()

from asymmetrize_test import asymmetrize_type_11_ms_cy

DEBUG_FLAG = False


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    ts_file = Path(r"P:\Synchronize\IWS\Testings\fourtrans_practice\iaaftsa\test_maiden_400\data_extracted\sim_data_00.csv")

    half_window_size = 200

    n_levels = 100
    max_shift_exp = 1.0
    max_shift = 20
    pre_vals_ratio = 0.95
    asymm_n_iters = 1
    prob_center = 0.0
    pre_val_exp = 1.0
    crt_val_exp = 1.0
    level_thresh_cnst = 100
    level_thresh_slp = -0.0
    rand_err_sclr_cnst = 0.0
    rand_err_sclr_rel = 0.00
    probs_exp = 0.0
    #==========================================================================

    if False:
    # if True:
        rel_dists = np.concatenate((
            np.arange(half_window_size, -1, -1.0,),
            np.arange(1.0, half_window_size + 1.0)))

        rel_dists /= rel_dists.max() + 1.0

        # kern_ftn = KERNEL_FTNS_DICT['epan']
        # kern_ftn = KERNEL_FTNS_DICT['sigm']
        # kern_ftn = KERNEL_FTNS_DICT['logi']
        # kern_ftn = KERNEL_FTNS_DICT['tric']
        kern_ftn = KERNEL_FTNS_DICT['triw']
        # kern_ftn = KERNEL_FTNS_DICT['biwe']

        ref_ser = kern_ftn(rel_dists)
        ref_ser /= ref_ser.max()

        # ref_ser = 1 - ref_ser

        ref_ser = np.round(np.concatenate((ref_ser, ref_ser)), 4)

    else:
        # ts_ser = pd.read_csv(ts_file, sep=';', index_col=0).loc['2000-01-01':'2001-12-31', 'q_obs']
        ts_ser = pd.read_csv(ts_file, sep=';', index_col=None).loc[:, '420']
        # ts_ser = pd.read_csv(ts_file, sep=';', index_col=None).loc[:, 'prec']

        ref_ser = ts_ser.values

        # probs = rankdata(ref_ser) / (ref_ser.size + 1.0)
        # ref_ser = expon.ppf(probs)

        # ref_ser[0] = -1

    probs = rankdata(ref_ser, axis=0) / (ref_ser.shape[0] + 1.0)

    # probs[probs < 0.6] = 0.6

    # probs = (ref_ser - ref_ser.min()) / (ref_ser.max() - ref_ser.min())

    # probs **= 0.5

    # ref_ser = expon.ppf(probs)

    inc = ref_ser.min()
    # inc = probs.min() * 1
    asymms_rand_err = np.random.random(ref_ser.shape[0])
    # asymms_rand_err *= 0

    # rand_err_cnst = -inc + (2 * inc) * np.random.random(ref_ser.shape[0])
    rand_err_rel = -0.5 + np.random.random(ref_ser.shape[0])
    rand_err_cnst = -inc + (2 * inc) * rand_err_rel

    ref_ser = np.round(ref_ser, 6).reshape(-1, 1).copy(order='f')
    probs = probs.reshape(-1, 1).copy(order='f')
    asymms_rand_err = asymms_rand_err.copy(order='f').reshape(-1, 1)

    rand_err_cnst = rand_err_cnst.copy(order='f').reshape(-1, 1)
    rand_err_rel = rand_err_rel.copy(order='f').reshape(-1, 1)

    ref_ser_srt = np.sort(ref_ser, axis=0)
    #==========================================================================

    n_levelss = np.array([n_levels], dtype=np.uint32)
    max_shift_exps = np.array([max_shift_exp], dtype=float)
    max_shifts = np.array([max_shift], dtype=np.int64)
    pre_vals_ratios = np.array([pre_vals_ratio], dtype=float)
    asymm_n_iterss = np.array([asymm_n_iters], dtype=np.uint32)
    prob_centers = np.array([prob_center], dtype=float)
    pre_val_exps = np.array([pre_val_exp], dtype=float)
    crt_val_exps = np.array([crt_val_exp], dtype=float)
    level_thresh_cnsts = np.array([level_thresh_cnst], dtype=np.int64)
    level_thresh_slps = np.array([level_thresh_slp], dtype=float)
    rand_err_sclrs_cnsts = np.array([rand_err_sclr_cnst], dtype=float)
    rand_err_sclrs_rels = np.array([rand_err_sclr_rel], dtype=float)
    probs_exps = np.array([probs_exp], dtype=float)
    #==========================================================================

    # asym_ser = asymmetrize_type_3_cy(
    #     # ref_ser,
    #     probs,
    #     probs,
    #     n_levels,
    #     max_shift_exp,
    #     max_shift,
    #     pre_vals_ratio,
    #     asymm_n_iters,
    #     asymms_rand_err,
    #     prob_center,
    #     )
    #
    # asym_ser_rev = asymmetrize_type_3_cy(
    #     # ref_ser,
    #     asym_ser,
    #     probs,
    #     n_levels,
    #     max_shift_exp,
    #     -max_shift,
    #     pre_vals_ratio,
    #     asymm_n_iters,
    #     asymms_rand_err,
    #     prob_center,
    #     )

    asym_ser = asymmetrize_type_11_ms_cy(
        # ref_ser,
        probs,
        probs,
        n_levelss,
        max_shift_exps,
        max_shifts,
        pre_vals_ratios,
        asymm_n_iterss,
        asymms_rand_err,
        prob_centers,
        pre_val_exps,
        crt_val_exps,
        level_thresh_cnsts,
        level_thresh_slps,
        rand_err_sclrs_cnsts,
        rand_err_sclrs_rels,
        rand_err_cnst,
        rand_err_rel,
        probs_exps)

    assert np.all(np.isfinite(asym_ser))

    # asym_ser_rev = asymmetrize_type_5_ms_cy(
    #     # ref_ser,
    #     asym_ser,
    #     probs,
    #     n_levelss,
    #     max_shift_exps,
    #     -max_shifts,
    #     pre_vals_ratios,
    #     asymm_n_iterss,
    #     asymms_rand_err,
    #     prob_centers,
    #     )

    plt.figure()

    # asym_ser = ref_ser

    for i in range(1):

        # beg_idx = np.argmin(asym_ser[:, i])
        # asym_ser = np.roll(asym_ser, -beg_idx, axis=0)
        #
        # assert np.argmin(asym_ser[:, i]) == 0

        asym_ser[:, i] = ref_ser_srt[np.argsort(np.argsort(asym_ser[:, i])), i]
        # asym_ser[:, i] += asym_ser[:, i] + (-0.1 + (0.2 * np.random.random(size=asym_ser.shape[0])))

        # asym_ser[:, i] = ref_ser_srt[np.argsort(np.argsort(asym_ser[:, i])), i]

        # asym_ser_rev[:, i] = ref_ser_srt[np.argsort(np.argsort(asym_ser_rev[:, i])), i]

        plt.plot(ref_ser[:, i], alpha=0.7, lw=2, c='r', label='ref')
        plt.plot(asym_ser[:, i], alpha=0.7, lw=1.5, c='b', label='asym')
        # plt.plot(asym_ser_rev[:, i], alpha=0.7, lw=1.5, c='g', label='asym_rev')

        # plt.plot(asymr_ser[:, i], alpha=0.7, lw=1.5, c='g', ls='-.', label='asymr')
        # plt.plot(probs[:, i], alpha=0.7, lw=1.5, c='k', ls='dotted', label='probs')

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.legend()

    plt.xlabel('Time step')
    plt.ylabel('Magnitude')

    plt.title(
        f'n_levels: {n_levels}, '
        f'max_shift_exp: {max_shift_exp:0.2f}, '
        f'max_shift: {max_shift}\n'
        f'pre_vals_ratio: {pre_vals_ratio:0.2f}, '
        f'asymm_n_iters: {asymm_n_iters}, '
        f'prob_center: {prob_center:0.2f}')

    plt.show(block=False)

    asym_ser_probs = rankdata(asym_ser[:, 0]) / (asym_ser.shape[0] + 1.0)
    # asym_ser_probs_rev = rankdata(asym_ser_rev[:, 0]) / (asym_ser_rev.shape[0] + 1.0)

    axes = plt.subplots(2, 2)[1].ravel()

    scatt_alpha = 0.2

    ofst = 1
    axes[0].scatter(asym_ser_probs[:-ofst], asym_ser_probs[ofst:], alpha=scatt_alpha, label='lag1')
    axes[0].grid()
    axes[0].set_axisbelow(True)
    axes[0].legend()
    axes[0].set_aspect('equal')
    axes[0].plot([0, 1], [0, 1], c='k', alpha=0.4)

    ofst += 1
    axes[1].scatter(asym_ser_probs[:-ofst], asym_ser_probs[ofst:], alpha=scatt_alpha, label='lag2')
    axes[1].grid()
    axes[1].set_axisbelow(True)
    axes[1].legend()
    axes[1].set_aspect('equal')
    axes[1].plot([0, 1], [0, 1], c='k', alpha=0.4)

    ofst += 1
    axes[2].scatter(asym_ser_probs[:-ofst], asym_ser_probs[ofst:], alpha=scatt_alpha, label='lag3')
    axes[2].grid()
    axes[2].set_axisbelow(True)
    axes[2].legend()
    axes[2].set_aspect('equal')
    axes[2].plot([0, 1], [0, 1], c='k', alpha=0.4)

    ofst += 1
    axes[3].scatter(asym_ser_probs[:-ofst], asym_ser_probs[ofst:], alpha=scatt_alpha, label='lag4')
    axes[3].grid()
    axes[3].set_axisbelow(True)
    axes[3].legend()
    axes[3].set_aspect('equal')
    axes[3].plot([0, 1], [0, 1], c='k', alpha=0.4)

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
