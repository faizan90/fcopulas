'''
@author: Faizan-Uni-Stuttgart

May 12, 2021

8:20:39 AM

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

# randint = np.random.randint
choice = np.random.choice
np.set_printoptions(edgeitems=10, linewidth=180)

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

#     dens = 1 / n_bins
#
#     etpy = -np.log(dens)

    etpy = 0.0 * n_bins

    return etpy


def _get_etpy_max(n_bins):

    dens = (1 / (n_bins ** 2))

    etpy = -np.log(dens)

    return etpy


def get_cond_u2_ecop_2d(u2s, u1, ecop_dens_arrs):

    ecop_u1_idx = int(u1 * ecop_dens_arrs.shape[0])

    ecop_ps = ecop_dens_arrs[ecop_u1_idx,:] * ecop_dens_arrs.shape[0]

    assert ecop_ps.sum()

    ecop_ps /= ecop_ps.sum()

    sel_idxs = np.where(ecop_ps)[0]

    ecop_u2_idx = np.random.choice(sel_idxs, p=ecop_ps[sel_idxs])

    idx2 = int(((ecop_u2_idx * (u2s.size + 0))) / ecop_dens_arrs.shape[0])

    ratio = u2s.size // ecop_dens_arrs.shape[0]
    idx2 += np.random.choice(np.arange(ratio))

    u2 = ((idx2 + 1) / (u2s.size + 1))

    return idx2, u2


def sample_from_ecop_2d(u1s, u2s, ecop_dens_arrs):

    nvs = u1s.size

    # A sampled realization.
    us = np.empty((nvs, 2), dtype=np.float64)

    i = 0
    smpled_idxs = []
    while i < nvs:
        if i == 0:
            # random idx.
            idx1 = nvs // 2  # choice(np.arange(0, nvs))
            u1 = ((idx1 + 1) / (u1s.size + 1))

            idx2, u2 = get_cond_u2_ecop_2d(u2s, u1, ecop_dens_arrs)

        else:
            idx1 = idx2
            u1 = u2

            idx2, u2 = get_cond_u2_ecop_2d(u2s, u1, ecop_dens_arrs)

        smpled_idxs.append(idx2)

        us[i,:] = u1, u2

        i += 1

    smpled_idxs = np.array(smpled_idxs)

    assert np.all(smpled_idxs >= 0) and np.all(smpled_idxs < u1s.size)

    unq_smpled_idxs, smpled_idxs_freq = np.unique(
        smpled_idxs, return_counts=True)

    freqs = np.zeros(u1s.size, dtype=np.float64)

    freqs[unq_smpled_idxs] = smpled_idxs_freq

#     return (rankdata(us))[::-1] / (nvs + 1), freqs
#     return us[::-1], freqs
#     return us, freqs
#     return (rankdata(us, axis=0))[::-1] / (nvs + 1), freqs
    return us[::-1], freqs


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\Copulas_practice\sample_from_ecop')
    os.chdir(main_dir)

    in_file = Path(r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    fig_size_a = (12, 8)
    fig_size_b = (15, 8)

    beg_time = '1991-01-01'
    end_time = '2000-12-31'

    stn = '420'

    suff = 'aj'

    out_name_a = f'{suff}_ecop_props.png'
    out_name_b = f'{suff}_ecops.png'
    out_name_c = f'{suff}_idxs_hist.png'
    out_name_d = f'{suff}_ecop_dens.png'

    in_ser = pd.read_csv(in_file, sep=';', index_col=0)[stn]

    in_ser.index = pd.to_datetime(in_ser.index, format='%Y-%m-%d')

    in_ser = in_ser.loc[beg_time:end_time]

    vals = in_ser.values + (-0.01 + (0.02 * np.random.random(in_ser.shape[0])))

    vals_sort = np.sort(vals)

    n_sims = 1009

    lag_steps_sample = 1

    lag_steps = np.arange(1, 31, dtype=np.int64)
    ecop_bins = vals.shape[0] // 4  # - lag_steps_sample
    ecop_rows = 3
    ecop_cols = 6
    ecop_lag_step = 1

    assert (ecop_rows * ecop_cols) <= (n_sims + 1), (
        'Not enough simulations!')

    main_fig_shape = (3, 2)

    fig_a = plt.figure(figsize=fig_size_a)

    fig_b, cop_axes = plt.subplots(
        ecop_rows,
        ecop_cols,
        sharex=True,
        sharey=True,
        figsize=fig_size_b)

    cop_axes = cop_axes.ravel()

    [cop_ax.set_aspect('equal') for cop_ax in cop_axes]

    plt.figure(fig_a.number)
    ts_ax = plt.subplot2grid(main_fig_shape, (0, 0), 1, 2)
    srho_ax = plt.subplot2grid(main_fig_shape, (1, 0), 1, 1)
    etpy_ax = plt.subplot2grid(main_fig_shape, (1, 1), 1, 1)
    aord_ax = plt.subplot2grid(main_fig_shape, (2, 0), 1, 1)
    adir_ax = plt.subplot2grid(main_fig_shape, (2, 1), 1, 1)

    # Time series.
    ts_ax.plot(
        vals,
        color='r',
        alpha=0.75,
        lw=1.5)

    ts_ax.grid()
    ts_ax.set_axisbelow(True)
    ts_ax.set_xlabel('Time step')
    ts_ax.set_ylabel('Discharge ($m^3.s^{-1}$)')

    etpy_min = _get_etpy_min(ecop_bins)
    etpy_max = _get_etpy_max(ecop_bins)

    ecop_dens_arrs = np.full((ecop_bins, ecop_bins), np.nan, dtype=np.float64)

    u1s = rankdata(
        vals[:-lag_steps_sample]) / (vals.shape[0] - lag_steps_sample + 1)

    u2s = rankdata(
        vals[+lag_steps_sample:]) / (vals.shape[0] - lag_steps_sample + 1)

    fill_bi_var_cop_dens(u1s, u2s, ecop_dens_arrs)

    ecop_dens_arrs_main = ecop_dens_arrs.copy()

    us_concate = np.concatenate(
        (u1s.reshape(-1, 1), u2s.reshape(-1, 1)),
        axis=1)

    phs_rand_sers = [[us_concate, None]]

    for _ in range(n_sims):
        phs_rand_sers.append(
            sample_from_ecop_2d(u1s, u2s, ecop_dens_arrs))

#     return

    sampled_idxs = []
    for j in reversed(range(n_sims + 1)):

        print(j)

        probs = phs_rand_sers[j][0]

        if not j:

            lclr = 'r'
            lalpha = 0.8
            lw = 1.5

        else:
            sampled_idxs.append(phs_rand_sers[j][1])

            lclr = 'k'
            lalpha = 0.2
            lw = 1

        if j == 1:
            ts_ax.plot(
                vals_sort[lag_steps_sample:][np.argsort(np.argsort(probs[:vals_sort[lag_steps_sample:].size, 0]))],
                color='k',
                alpha=0.5,
                lw=1)

        scorrs = []
        asymms_1 = []
        asymms_2 = []
        etpys = []
        pcorrs = []
        for lag_step in lag_steps:
            probs_i, rolled_probs_i = roll_real_2arrs(
                probs[:, 0], probs[:, 0], lag_step, True)

            new_data = vals_sort[
                lag_steps_sample:][np.argsort(np.argsort(probs_i[:vals_sort[lag_steps_sample:].size]))]

            data_i, rolled_data_i = roll_real_2arrs(
                new_data, new_data, lag_step)

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

            if lag_step != ecop_lag_step:
                continue

            if j >= (ecop_rows * ecop_cols):
                continue

            plt.figure(fig_b.number)

            cop_axes[j].scatter(
                probs_i, rolled_probs_i, color='r', alpha=20 / 255)

            cop_axes[j].grid()

            if not (j % ecop_cols):
                cop_axes[j].set_ylabel('$u_2$')

            if (j // ecop_cols) == (ecop_rows - 1):
                cop_axes[j].set_xlabel('$u_1$')

        plt.figure(fig_a.number)

        # Spearman rho.
        srho_ax.plot(
            lag_steps,
            scorrs,
            color=lclr,
            alpha=lalpha,
            lw=lw)

        # Entropy.
        etpy_ax.plot(
            lag_steps,
            etpys,
            color=lclr,
            alpha=lalpha,
            lw=lw)

        # Order asymmetry.
        aord_ax.plot(
            lag_steps,
            asymms_1,
            color=lclr,
            alpha=lalpha,
            lw=lw)

        # Directional asymmetry.
        adir_ax.plot(
            lag_steps,
            asymms_2,
            color=lclr,
            alpha=lalpha,
            lw=lw)

    mean_freqs = np.array(sampled_idxs).mean(axis=0)
#     mean_freqs = np.median(np.array(sampled_idxs), axis=0)
#     mean_freqs = np.max(np.array(sampled_idxs), axis=0)
    print(mean_freqs.min(), mean_freqs.max())
    print(np.where(mean_freqs == 0)[0])
    print(mean_freqs.size)
    print(mean_freqs[:+20])
    print(mean_freqs[-20:])

    plt.figure(fig_a.number)

    srho_ax.grid()
    srho_ax.set_axisbelow(True)
    srho_ax.set_ylabel('$\\rho_s$')

    etpy_ax.grid()
    etpy_ax.set_axisbelow(True)
    etpy_ax.set_ylabel('$H$')

    aord_ax.grid()
    aord_ax.set_axisbelow(True)
    aord_ax.set_xlabel('Lead step')
    aord_ax.set_ylabel('$AO$')

    adir_ax.grid()
    adir_ax.set_axisbelow(True)
    adir_ax.set_xlabel('Lead step')
    adir_ax.set_ylabel('$AD$')

    # Save.
    plt.figure(fig_a.number)

    plt.tight_layout()
    plt.savefig(str(out_name_a), bbox_inches='tight', dpi=300)

    plt.figure(fig_b.number)

    plt.tight_layout()
    plt.savefig(str(out_name_b), bbox_inches='tight', dpi=300)

    plt.close('all')

    # sampled_idxs_hist.

    plt.figure()
    plt.bar(np.arange(u1s.size), mean_freqs)
#     plt.tight_layout()
    plt.savefig(str(out_name_c), bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure()

#     pcolmesh_x_coords = np.linspace(-0.5, ecop_bins - 0.5, (ecop_bins + 1))
#
#     pcolmesh_y_coords = np.linspace(-0.5, ecop_bins - 0.5, (ecop_bins + 1))
#
#     interp_x_crds_plt_msh, interp_y_crds_plt_msh = (
#         np.meshgrid(pcolmesh_x_coords, pcolmesh_y_coords))
#
#     plt.pcolormesh(interp_x_crds_plt_msh, interp_y_crds_plt_msh, ecop_dens_arrs)

    plt.imshow(ecop_dens_arrs_main * u1s.size, origin='lower')
    plt.grid()
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.savefig(str(out_name_d), bbox_inches='tight', dpi=300)
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