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

from fcopulas import (
    sample_from_2d_ecop,
    get_asymms_sample,
    fill_bi_var_cop_dens,
    get_asymm_1_max,
    get_asymm_2_max,
    get_etpy_min,
    get_etpy_max)

from phsann.misc import roll_real_2arrs

# randint = np.random.randint
choice = np.random.choice
np.set_printoptions(edgeitems=10, linewidth=180)

plt.ioff()

DEBUG_FLAG = False


def get_kernel_ecop_dens(ecop_freqs_arr):

    if False:
        kern_ecop_arr = ecop_freqs_arr.copy()
        n_bins = ecop_freqs_arr.shape[0]

        for i in range(1, n_bins - 1):
            for j in range(1, n_bins - 1):
                efreq = ecop_freqs_arr[i, j]

                if not efreq:
                    continue

                kern_ecop_arr[i - 1:i + 2, j - 1:j + 2] += efreq * 0.25

        for i in range(2, n_bins - 2):
            for j in range(2, n_bins - 2):
                efreq = ecop_freqs_arr[i, j]

                if not efreq:
                    continue

                kern_ecop_arr[i - 2:i + 3, j - 2:j + 3] += efreq * 0.25

        kern_ecop_arr /= kern_ecop_arr.sum()

    elif False:
        # Gaussian, tri-variate kernel.
        n_bins = ecop_freqs_arr.shape[0]
        kern_ecop_arr = np.zeros((n_bins + 2, n_bins + 2))

        kern = (1 / 16) * np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
            ], dtype=float)

        for i in range(0, n_bins):
            for j in range(0, n_bins):
                efreq = ecop_freqs_arr[i, j]

                if not efreq:
                    continue

                kern_ecop_arr[i:i + 3, j:j + 3] += kern * efreq

        kern_ecop_arr = kern_ecop_arr[1:-1, 1:-1].copy(order='c')
        kern_ecop_arr /= kern_ecop_arr.sum()

    elif False:
        # Gaussian, pent-variate kernel.
        n_bins = ecop_freqs_arr.shape[0]
        kern_ecop_arr = np.zeros((n_bins + 4, n_bins + 4))

        kern = (1 / 256) * np.array([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
            ], dtype=float)

        for i in range(0, n_bins):
            for j in range(0, n_bins):
                efreq = ecop_freqs_arr[i, j]

                if not efreq:
                    continue

                kern_ecop_arr[i:i + 5, j:j + 5] += kern * efreq

        kern_ecop_arr = kern_ecop_arr[2:-2, 2:-2].copy(order='c')
        kern_ecop_arr /= kern_ecop_arr.sum()

    else:
        kern_ecop_arr = ecop_freqs_arr / ecop_freqs_arr.sum()

    return kern_ecop_arr


def get_cond_u2_ecop_2d(nvs, u1, ecop_dens_arr, idxs_freqs):

    ecop_u1_idx = int(u1 * ecop_dens_arr.shape[0])

#     ecop_ps = ecop_dens_arr[ecop_u1_idx,:] * ecop_dens_arr.shape[0]
    ecop_ps = ecop_dens_arr[:, ecop_u1_idx] * ecop_dens_arr.shape[0]

#     ecop_ps = np.round(
#         ecop_dens_arr[:, ecop_u1_idx] * ecop_dens_arr.shape[0], 0)

    assert ecop_ps.sum()

    # In case nvs is not a multipe of ecop_bins.
#     ecop_ps /= ecop_ps.sum()

    sel_idxs = np.where(ecop_ps)[0]

    ratio = nvs // ecop_dens_arr.shape[0]

    new_idxs = []
    ecops_pss = []
    for idx, ecop_p in zip(sel_idxs, ecop_ps[sel_idxs]):

        beg = int((idx * nvs) / ecop_dens_arr.shape[0])
        end = beg + ratio

        new_idxs.extend(list(range(beg, end)))
        ecops_pss.extend([ecop_p / ratio] * ratio)

    new_idxs = np.array(new_idxs)

    if np.all(idxs_freqs[new_idxs]):
#         raise ValueError('All ones!')
        zero_idxs = np.where(idxs_freqs == 0)[0]

        # Fully random selection of next index.
#         idx2 = np.random.choice(zero_idxs)

        # Nearest that is zero.
        min_diffs = np.full(zero_idxs.size, np.inf)
        min_diff_idxs = np.full(zero_idxs.size, zero_idxs.size, dtype=int)
        for i, zero_idx in enumerate(zero_idxs):
            diffs = ((zero_idx - new_idxs) ** 2)
            min_diff = diffs.min()

            if min_diff < min_diffs[i]:
                min_diffs[i] = min_diff
                min_diff_idxs[i] = np.argmin(diffs)

        idx2 = zero_idxs[np.argmin(min_diffs)]

    else:
        sample_idxs = np.where(idxs_freqs[new_idxs] == 0)[0]

        ecops_pss = np.array(ecops_pss)[sample_idxs]

        ecops_pss /= ecops_pss.sum()

#         ctr = 0
        idx2 = np.random.choice(new_idxs[sample_idxs], p=ecops_pss)

#         idx2 = new_idxs[sample_idxs][nrst_idx - 1]

#         while idxs_freqs[idx2] > 0:
#             idx2 = np.random.choice(new_idxs, p=ecops_pss)
#
#             ctr += 1
#
#             if ctr == nvs * 100:
#                 raise Exception('Huh?')

#     ecop_u2_idx = np.random.choice(sel_idxs, p=ecop_ps[sel_idxs])
# #     ecop_u2_idx = np.random.choice(sel_idxs)
#
#     idx2 = int(((ecop_u2_idx * (nvs + 0))) / ecop_dens_arr.shape[0])
#
#     ratio_rng = np.arange(ratio)
#
#     rand_idx = np.random.choice(ratio_rng)
#     while (idx2 + rand_idx) == idx1:
#         rand_idx = np.random.choice(ratio_rng)
#
#     idx2 += rand_idx

    u2 = ((idx2 + 1) / (nvs + 1))

    assert 0 < u2 < 1, u2

    return idx2, u2


def sample_from_ecop_2d(ecop_dens_arr, nvs, sim_no):

    print('sim:', sim_no)

    # A sampled realization.
    us = np.empty(nvs, dtype=np.float64)

    idxs_freqs = np.zeros(nvs, dtype=np.uint64)

    smpled_idxs = np.zeros(nvs, dtype=np.float64)

    # random idx.
    idx1 = sim_no  # nvs // 2  #  nvs - 1  # choice(np.arange(0, nvs))  #
    u1 = ((idx1 + 1) / (nvs + 1))

    idxs_freqs[idx1] += 1  # before entering get_cond_u2_ecop_2d.
    smpled_idxs[idx1] += 1

    us[0] = u1

    i = 1
    while i < nvs:
        idx2, u2 = get_cond_u2_ecop_2d(nvs, u1, ecop_dens_arr, idxs_freqs)

#         idx2, u2 = get_cond_idx2_2d_ecop(ecop_dens_arr, idxs_freqs, u1)

#         print(idx2, u2)
#         print(get_cond_idx2_2d_ecop(ecop_dens_arr, idxs_freqs, u1))
#         raise Exception
        idx1 = idx2
        u1 = u2

        idxs_freqs[idx1] += 1
        smpled_idxs[idx1] += 1

#         if i == (nvs - 1):
#             pass
#
#         else:
#             idx2, u2 = get_cond_u2_ecop_2d(nvs, u1, ecop_dens_arr, idxs_freqs)

        us[i] = u1

        i += 1

    assert np.all(idxs_freqs == 1)

    return us, smpled_idxs


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\Copulas_practice\sample_from_ecop')
    os.chdir(main_dir)

    in_file = Path(r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    fig_size_a = (12, 8)
    fig_size_b = (15, 8)

    beg_time = '1997-01-01'
    end_time = '2000-12-31'

    stn = '420'

    suff = 'ax'

    out_name_a = f'{suff}_ecop_props.png'
    out_name_b = f'{suff}_ecops.png'
    out_name_c = f'{suff}_idxs_hist.png'
    out_name_d = f'{suff}_ecop_dens.png'
    out_name_e = f'{suff}_probs_hist.png'

    in_ser = pd.read_csv(in_file, sep=';', index_col=0)[stn]

    in_ser.index = pd.to_datetime(in_ser.index, format='%Y-%m-%d')

    in_ser = in_ser.loc[beg_time:end_time]

    np.random.seed((2 ** 32) - 1)
    vals = in_ser.values + (-0.01 + (0.02 * np.random.random(in_ser.shape[0])))

    vals_sort = np.sort(vals)

#     n_sims = 20

    lag_steps_sample = 1

    nvs = vals.shape[0] - lag_steps_sample

    n_sims = 20

    lag_steps = np.arange(1, 31, dtype=np.int64)
    ecop_bins = (vals.shape[0] // 20)  # - lag_steps_sample
    ecop_rows = 3
    ecop_cols = 6
    ecop_lag_step = 15

    assert not ((vals.size - lag_steps_sample) % ecop_bins), (
        'Length of input not a multiple of ecop_bins!')

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

    etpy_min = get_etpy_min(ecop_bins)
    etpy_max = get_etpy_max(ecop_bins)

    ecop_dens_arr = np.full((ecop_bins, ecop_bins), np.nan, dtype=np.float64)

    u1s = rankdata(
        vals[:-lag_steps_sample]) / (vals.shape[0] - lag_steps_sample + 1)

    u2s = rankdata(
        vals[+lag_steps_sample:]) / (vals.shape[0] - lag_steps_sample + 1)

    fill_bi_var_cop_dens(u1s, u2s, ecop_dens_arr)
    ecop_dens_arr = get_kernel_ecop_dens(ecop_dens_arr * u1s.size)

    ecop_dens_arr_main = ecop_dens_arr.copy()

    phs_rand_sers = [[u1s, None]]

    beg_t = timeit.default_timer()
    for sim_no in range(n_sims):
#         phs_rand_sers.append(
#             sample_from_ecop_2d(ecop_dens_arr, nvs, sim_no))

        phs_rand_sers.append(
            sample_from_2d_ecop(ecop_dens_arr, nvs, sim_no))

    print('Took ', timeit.default_timer() - beg_t)
#     raise Exception

#     return

    sampled_idxs = []
    for j in reversed(range(n_sims + 1)):
#     for j in range(30):

        print('sim_no:', j)

        probs = phs_rand_sers[j][0]
        print('unqs:', np.unique(probs).size)

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
                vals_sort[lag_steps_sample:][np.argsort(np.argsort(probs))],
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
                probs, probs, lag_step, True)

            new_data = vals_sort[
                lag_steps_sample:][np.argsort(np.argsort(probs_i[:vals_sort[lag_steps_sample:].size]))]

            data_i, rolled_data_i = roll_real_2arrs(
                new_data, new_data, lag_step)

            # scorr.
            scorr = np.corrcoef(probs_i, rolled_probs_i)[0, 1]
            scorrs.append(scorr)

            # asymms.
            asymm_1, asymm_2 = get_asymms_sample(probs_i, rolled_probs_i)

            asymm_1 /= get_asymm_1_max(scorr)

            asymm_2 /= get_asymm_2_max(scorr)

            asymms_1.append(asymm_1)
            asymms_2.append(asymm_2)

            # ecop etpy.
            fill_bi_var_cop_dens(probs_i, rolled_probs_i, ecop_dens_arr)
            ecop_dens_arr = get_kernel_ecop_dens(ecop_dens_arr * u1s.size)

            non_zero_idxs = ecop_dens_arr > 0

            dens = ecop_dens_arr[non_zero_idxs]

            etpy_arr = -(dens * np.log(dens))

            etpy = etpy_arr.sum()

            etpy = (etpy - etpy_min) / (etpy_max - etpy_min)

#             etpy = 0.5
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

    print('Plotting props...')
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

    print('Plotting ecops...')
    plt.figure(fig_b.number)

    plt.tight_layout()
    plt.savefig(str(out_name_b), bbox_inches='tight', dpi=300)

    plt.close('all')

    print('Plotting sampled idxs histogram...')
    plt.figure()
    plt.bar(np.arange(u1s.size), mean_freqs)
    plt.savefig(str(out_name_c), bbox_inches='tight', dpi=300)
    plt.close()

    print('Plotting ecop histogram...')
    plt.figure()
    plt.imshow(ecop_dens_arr_main * u1s.size, origin='lower')
    plt.grid()
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.savefig(str(out_name_d), bbox_inches='tight', dpi=300)
    plt.close()

    print('Plotting ref vs sim0 probs histogram...')
    axes = plt.subplots(1, 2, squeeze=False)[1]

    axes[0, 0].hist(
        phs_rand_sers[0][0], bins=20, range=(0, 1), rwidth=0.8)

    axes[0, 1].hist(
        phs_rand_sers[1][0], bins=20, range=(0, 1), rwidth=0.8)

    plt.savefig(str(out_name_e), bbox_inches='tight', dpi=300)
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
