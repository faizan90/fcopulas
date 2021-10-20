'''
@author: Faizan-Uni-Stuttgart

Oct 15, 2021

4:10:58 PM

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
from scipy.optimize import differential_evolution

from ana_sample_from_sid_tcop import (
    get_sid_tcop_probs_cross,
    get_sid_tcop_probs_auto,
    get_copula_props_lags,
    plot_props)

DEBUG_FLAG = False

np_rand = np.random.random

np.set_printoptions(
    threshold=2000,
    linewidth=200000,
    formatter={'float': '{:+9.4f}'.format})


def obj_ftn_fit_sidd_tcop_prms(
        prms, ref_props, n_bins, wts, cop_type_ftn, max_lag):

    sim_probs = cop_type_ftn(prms).copy(order='c')

    sim_props = get_copula_props_lags(sim_probs, max_lag, n_bins)

    sq_diff_sum = ((((ref_props - sim_props) / ref_props) ** 2) * wts).sum()

    print(f'sq_diff_sum: {sq_diff_sum:20.6f}')
    # print(f'ref_props: {ref_props}')
    # print(f'sim_props: {sim_props}')
    return sq_diff_sum


def obj_ftn_fit_sidd_tcop_cop(prms, ref_probs_srtd, cop_type_ftn):

    sim_probs = cop_type_ftn(prms)

    sim_probs_srtd = sim_probs[np.argsort(sim_probs[:, 0]),:]

    sq_diff_sum = ((ref_probs_srtd[:, 1:] - sim_probs_srtd[:, 1:]) ** 2).sum()

    print(f'sq_diff_sum: {sq_diff_sum:20.6f}')
    return sq_diff_sum

# def obj_ftn_fit_sidd_tcop_dcop(prms, ref_hist, n_bins, wts, cop_type_ftn):
#
#     sim_probs = cop_type_ftn(prms).copy(order='c')
#
#     sim_hist = get_hist_nd(sim_probs, n_bins)
#
#     sq_diff_sum = (((ref_hist - sim_hist) ** 2) * wts).sum() / float(
#         sim_probs.shape[1])
#
#     print(f'sq_diff_sum: {sq_diff_sum:20.6f}')
#     return sq_diff_sum


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\Copulas_practice\tcop')
    os.chdir(main_dir)

    # in_data_file = Path(
    #     r'neckar_full_neckar_avg_temp_kriging_1961-01-01_to_2015-12-31_1km_all__EDK.csv')

    in_data_file = Path(
        r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    read_col = '454'

    # Copula stuff.
    n_bins = 20

    max_lag = 20

    wts_prms = np.array([1.0, 5.0, 5.0, 0.0])

    shift_prms = np.array([+0.001, -0.001, -0.001, -0.001])

    # Optimization parameters.
    n_cpus = 8
    max_iters = 100
    opt_tol = 0.01

    # Copula type to evaluate.
    # cop_type = 'cross'
    cop_type = 'auto'

    # Type of the properties used to evaluate the best fit.
    fit_type = 'prms'  # Ecop prms (scorr, asymm1, asymm2, etyp).
    # fit_type = 'cop'  # Empirical cop.
    # fit_type = 'dcop'  # Discrete empirical cop.
    #==========================================================================

    in_df = pd.read_csv(in_data_file, sep=';', index_col=0)[read_col]

    in_data = in_df.values.reshape(-1, 1)

    n_steps = in_data.shape[0]

    in_probs = (rankdata(in_data, axis=0) / (n_steps + 1.0)).copy(order='c')

    in_props = get_copula_props_lags(in_probs, max_lag, n_bins)

    in_props_shift = in_props + shift_prms

    in_probs_srtd = in_probs[np.argsort(in_probs[:, 0], kind='mergesort'),:]

    in_pcorr = np.corrcoef(in_data[:-1, 0], in_data[1:, 0])[0, 1]

    prms_bds = np.array([
        [in_pcorr - 1e-4, 1.0 - 1e-5],
        [-5, +5],
        [-5, +5],
        [2.0, 500],
        [2.0, 500],
        [n_steps, n_steps],
        [-50, 50],
        [-50, 50],
        ], dtype=np.float64)

    beg_opt_time = timeit.default_timer()

    if cop_type == 'cross':
        cop_type_ftn = get_sid_tcop_probs_cross

    elif cop_type == 'auto':
        cop_type_ftn = get_sid_tcop_probs_auto

    else:
        raise ValueError(f'Unknown cop_type: {cop_type}!')

    if fit_type == 'prms':
        opt_ress = differential_evolution(
            obj_ftn_fit_sidd_tcop_prms,
            bounds=prms_bds,
            args=(in_props_shift, n_bins, wts_prms, cop_type_ftn, max_lag),
            updating='deferred',
            workers=n_cpus,
            maxiter=max_iters,
            tol=opt_tol)

    elif fit_type == 'cop':
        opt_ress = differential_evolution(
            obj_ftn_fit_sidd_tcop_cop,
            bounds=prms_bds,
            args=(in_probs_srtd, cop_type_ftn),
            updating='deferred',
            workers=n_cpus,
            maxiter=max_iters,
            tol=opt_tol)

    # elif fit_type == 'dcop':
    #     wts_dcop = 1.0 / in_hist
    #     wts_dcop[~np.isfinite(wts_dcop)] = 0.0
    #     wts_dcop **= 3.0
    #
    #     opt_ress = differential_evolution(
    #         obj_ftn_fit_sidd_tcop_dcop,
    #         bounds=prms_bds,
    #         args=(in_hist, n_bins, wts_dcop, cop_type_ftn),
    #         updating='deferred',
    #         workers=n_cpus,
    #         maxiter=max_iters,
    #         tol=opt_tol)

    else:
        raise ValueError(f'Unknown fit_type: {fit_type}!')

    end_opt_time = timeit.default_timer()

    print(f'Optimization took: {end_opt_time - beg_opt_time:0.3f} secs.')

    prms = opt_ress.x
    fun = opt_ress.fun
    #==========================================================================

    print(fun, prms)

    if fit_type == 'prms':
        print(obj_ftn_fit_sidd_tcop_prms(
            prms, in_props, n_bins, wts_prms, cop_type_ftn, max_lag))

    elif fit_type == 'cop':
        print(obj_ftn_fit_sidd_tcop_cop(prms, in_probs_srtd, cop_type_ftn))

    # elif fit_type == 'dcop':
    #     print(obj_ftn_fit_sidd_tcop_dcop(
    #         prms, in_hist, n_bins, wts_dcop, cop_type_ftn))

    else:
        raise ValueError(f'Unknown fit_type: {fit_type}!')

    sim_probs = cop_type_ftn(prms)

    sim_props = get_copula_props_lags(sim_probs, max_lag, n_bins)

    print(in_props)
    print(sim_props)

    ax_ref, ax_sim = plt.subplots(1, 2)[1]

    # Reference.
    ax_ref.scatter(
        in_probs[:-1, 0], in_probs[1:, 0], alpha=5 / 255.0)

    ax_ref.grid()
    ax_ref.set_axisbelow(True)

    ax_ref.set_title('Reference')

    ax_ref.set_aspect('equal')

    # Simulation.
    ax_sim.scatter(
        sim_probs[:-1, 0], sim_probs[1:, 0], alpha=5 / 255.0)

    ax_sim.grid()
    ax_sim.set_axisbelow(True)

    ax_sim.set_title('Simulation')

    ax_sim.set_aspect('equal')

    plt.show()
    plt.close()
    #==========================================================================

    # Temporal props.
    axes = plt.subplots(2, 3, squeeze=False, figsize=None)[1]

    plot_props(in_probs.reshape(-1, 1), 20, n_bins, axes, 'r', 'ref')
    plot_props(sim_probs.reshape(-1, 1), 20, n_bins, axes, 'b', 'sim')

    # For now.
    axes[0, 2].set_axis_off()
    axes[1, 2].set_axis_off()

    axes[0, 0].grid()
    axes[1, 0].grid()
    axes[1, 1].grid()
    axes[0, 1].grid()
    # axes[0, 2].grid()
    # axes[1, 2].grid()

    axes[0, 0].legend()
    axes[1, 0].legend()
    axes[1, 1].legend()
    axes[0, 1].legend()
    # axes[0, 2].legend()
    # axes[1, 2].legend()

    axes[0, 0].set_ylabel('Spearman correlation')

    axes[1, 0].set_xlabel('Lag steps')
    axes[1, 0].set_ylabel('Asymmetry (Type - 1)')

    axes[1, 1].set_xlabel('Lag steps')
    axes[1, 1].set_ylabel('Asymmetry (Type - 2)')

    axes[0, 1].set_ylabel('Entropy')

    # axes[0, 2].set_xlabel('Lag steps')
    # axes[0, 2].set_ylabel('Pearson correlation')

    plt.show()
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
