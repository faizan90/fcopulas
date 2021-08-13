'''
@author: Faizan-Uni-Stuttgart

Mar 9, 2021

8:09:08 PM

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

plt.ioff()

plt.rcParams.update({'font.size':16})

DEBUG_FLAG = False


def get_rft(data):

    assert data.ndim == 2
    assert data.size

    ft = np.fft.rfft(data, axis=0)

    return ft


def get_mag_spec(ft):

    mag_spec = np.abs(ft)

    return mag_spec


def get_phs_spec(ft):

    phs_spec = np.angle(ft)

    return phs_spec


def get_cumm_pow_spec(ref_data, sim_data):

    ref_ft = get_rft(ref_data)[1:,:]
    sim_ft = get_rft(sim_data)[1:,:]

    ref_mag = get_mag_spec(ref_ft)
    ref_phs = get_phs_spec(ref_ft)

    sim_mag = get_mag_spec(sim_ft)
    sim_phs = get_phs_spec(sim_ft)

    numr = (
        ref_mag[1:,:] *
        sim_mag[1:,:] *
        np.cos(ref_phs[1:,:] - sim_phs[1:,:]))

    demr = (
        ((ref_mag[1:,:] ** 2).sum(axis=0) ** 0.5) *
        ((sim_mag[1:,:] ** 2).sum(axis=0) ** 0.5))

    return np.cumsum(numr, axis=0) / demr


def get_cumm_phs_spec(ref_data, sim_data):

    ref_ft = get_rft(ref_data)[1:,:]
    sim_ft = get_rft(sim_data)[1:,:]

    ref_phs = get_phs_spec(ref_ft)
    sim_phs = get_phs_spec(sim_ft)

    numr = (np.cos(ref_phs[1:,:] - sim_phs[1:,:]))

    demr = ref_ft.shape[0]

#     demr = (
#         ((ref_mag[1:,:] ** 2).sum(axis=0) ** 0.5) *
#         ((sim_mag[1:,:] ** 2).sum(axis=0) ** 0.5))

    return np.cumsum(numr, axis=0) / demr


def get_cumm_mag_spec(ref_data, sim_data):

    ref_ft = get_rft(ref_data)[1:,:]
    sim_ft = get_rft(sim_data)[1:,:]

    ref_mag = get_mag_spec(ref_ft)
    sim_mag = get_mag_spec(sim_ft)

    numr = (
        ref_mag[1:,:] *
        sim_mag[1:,:])

    demr = (
        ((ref_mag[1:,:] ** 2).sum(axis=0) ** 0.5) *
        ((sim_mag[1:,:] ** 2).sum(axis=0) ** 0.5))

    return np.cumsum(numr, axis=0) / demr


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Projects\2016_DFG_SPATE\Meetings_Second_Phase\20210311_Online')

    os.chdir(main_dir)

    ref_files = [
        Path(r"P:\Synchronize\IWS\QGIS_Neckar\spate_cmpr__ppt_obs__monthly_scale_calib\03_hbv_figs\kf_01_calib_HBV_sim_420.csv"),
        Path(r"P:\Synchronize\IWS\QGIS_Neckar\spate_cmpr__ppt_cosmo__monthly_scale_calib\03_hbv_figs\kf_01_calib_HBV_sim_420.csv"),
        ]

    ref_cols = ['q_obs', 'q_obs']
    ref_labs = ['obs', 'obs']

    sim_files = [
        Path(r"P:\Synchronize\IWS\QGIS_Neckar\spate_cmpr__ppt_obs__monthly_scale_calib\03_hbv_figs\kf_01_calib_HBV_sim_420.csv"),
        Path(r"P:\Synchronize\IWS\QGIS_Neckar\spate_cmpr__ppt_cosmo__monthly_scale_calib\03_hbv_figs\kf_01_calib_HBV_sim_420.csv"),
        ]

    sim_cols = ['q_sim', 'q_sim']
    sim_labs = ['dwd', 'cosmo']

    fig_size = (15, 10)
    dpi = 300

    off_idx = 365
    sep = ';'
    plot_first_ref_only_flag = True

    ref_data = None
    for i, (ref_file, ref_col) in enumerate(zip(ref_files, ref_cols)):

        ref_ser = pd.read_csv(
            ref_file, sep=sep)[ref_col].iloc[off_idx:]

        if ref_data is None:
            ref_data = np.empty(
                (ref_ser.size, len(ref_files)))

        ref_data[:, i] = ref_ser.values

    sim_data = None
    for i, (sim_file, sim_col) in enumerate(zip(sim_files, sim_cols)):

        sim_ser = pd.read_csv(
            sim_file, sep=sep)[sim_col].iloc[off_idx:]

        if sim_data is None:
            sim_data = np.empty(
                (sim_ser.size, len(sim_files)))

        sim_data[:, i] = sim_ser.values

    assert sim_data.shape == ref_data.shape

#     ref_ref_cum_pow_spec = get_cumm_pow_spec(ref_data, ref_data)
#     ref_sim_cum_pow_spec = get_cumm_pow_spec(ref_data, sim_data)

#     ref_ref_cum_pow_spec = get_cumm_phs_spec(ref_data, ref_data)
#     ref_sim_cum_pow_spec = get_cumm_phs_spec(ref_data, sim_data)

    ref_ref_cum_pow_spec = get_cumm_mag_spec(ref_data, ref_data)
    ref_sim_cum_pow_spec = get_cumm_mag_spec(ref_data, sim_data)

    periods = (ref_data.shape[0]) / (
        np.arange(1, ref_data.shape[0] // 2))

    assert periods.size == ref_ref_cum_pow_spec.shape[0]

    plt.figure(figsize=fig_size)
    for i in range(ref_data.shape[1]):

        if plot_first_ref_only_flag and i:
            pass

        else:
            plt.semilogx(
                periods,
                ref_ref_cum_pow_spec[:, i],
                alpha=0.75,
                lw=3,
                label=f'{ref_labs[i]}-{ref_labs[i]}')

        plt.semilogx(
            periods,
            ref_sim_cum_pow_spec[:, i],
            alpha=0.6,
            lw=2,
            label=f'{ref_labs[i]}-{sim_labs[i]}')

    plt.legend()
    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.xlabel('Period (days)')
    plt.ylabel(f'Cummulative correlation contribution')

    plt.xlim(plt.xlim()[::-1])

#     plt.show()

    plt.savefig('hydmod_mag_corr_cmpr.png', bbox_inches='tight', dpi=dpi)
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
