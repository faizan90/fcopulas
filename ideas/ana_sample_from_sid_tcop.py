'''
@author: Faizan-Uni-Stuttgart

Oct 15, 2021

4:05:22 PM

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
from numpy.random import random as np_rand
from scipy.stats import invgamma, rankdata, norm
from numpy.random import normal as np_norm
import matplotlib.pyplot as plt; plt.ioff()
from numpy.random import multivariate_normal as mvn

from fcopulas import (
    get_asymm_1_sample,
    get_asymm_2_sample,
    get_asymm_1_max,
    get_asymm_2_max,
    get_hist_nd,
    get_etpy_nd,
    get_etpy_max_nd,
    get_etpy_min_nd)

DEBUG_FLAG = False


def lag_var(in_arr, fin_lag):

    assert isinstance(in_arr, np.ndarray)
    assert in_arr.ndim == 2

    assert isinstance(fin_lag, int)
    assert fin_lag >= 0

    assert in_arr.shape[1]
    assert in_arr.shape[0] > fin_lag

    if not fin_lag:
        out_arr = in_arr.copy()

    else:
        out_arr = []

        for i in range(fin_lag + 1):
            out_arr.append(in_arr[i:-(fin_lag + 1 - i)])

        out_arr = np.concatenate(out_arr, axis=1)

    return out_arr


def get_copula_props(probs, n_bins):

    probs = probs.copy(order='c')

    u = probs[:, 0].copy(order='c')
    v = probs[:, 1].copy(order='c')

    scorr = np.corrcoef(u, v)[0, 1]

    asymm1 = get_asymm_1_sample(u, v)
    asymm2 = get_asymm_2_sample(u, v)

    if (-1.0 + 1e-7) < scorr < (1.0 - 1e-7):
        asymm1 /= get_asymm_1_max(scorr)
        asymm2 /= get_asymm_2_max(scorr)

    etpy_nd_min = get_etpy_min_nd(n_bins, 2)
    etpy_nd_max = get_etpy_max_nd(n_bins, 2)

    hist_nd = get_hist_nd(probs, n_bins)

    etpy_nd = (get_etpy_nd(hist_nd, probs.shape[0]) - etpy_nd_min) / (
        etpy_nd_max - etpy_nd_min)

    return np.array([scorr, asymm1, asymm2, etpy_nd])


def get_copula_props_lags(probs, max_lag, n_bins):

    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)

    lagged_probs = lag_var(probs, max_lag)

    lagged_probs = rankdata(lagged_probs, axis=0) / (
        lagged_probs.shape[0] + 1.0)

    props = np.empty((lagged_probs.shape[1] - 1, 4))
    for i in range(1, lagged_probs.shape[1]):
        lag_probs = np.concatenate(
            (lagged_probs[:, 0].reshape(-1, 1),
             lagged_probs[:, i].reshape(-1, 1)), axis=1)

        props[i - 1] = get_copula_props(lag_probs, n_bins)

    return props


def plot_props(probs, max_lag, n_bins, axes, clr, label):

    plt_alpha = 0.9

    props = get_copula_props_lags(probs, max_lag, n_bins)

    lag_steps = np.arange(1, max_lag + 1)

    # axes = plt.subplots(2, 3, squeeze=False, figsize=fig_size)[1]

    # plot
    axes[0, 0].plot(
        lag_steps, props[:, 0], alpha=plt_alpha, color=clr, label=label)

    axes[1, 0].plot(
        lag_steps, props[:, 1], alpha=plt_alpha, color=clr, label=label)

    axes[1, 1].plot(
        lag_steps, props[:, 2], alpha=plt_alpha, color=clr, label=label)

    axes[0, 1].plot(
        lag_steps, props[:, 3], alpha=plt_alpha, color=clr, label=label)

    # axes[0, 2].plot(
    #     lag_steps, props[:, 4], alpha=plt_alpha, color=clr, label=label)

    return


def get_sid_tcop_probs_cross(args):

    pcorr, mean, cov, nu1, nu2, N, Gs0, Gs1 = args

    # Due to optimization.
    N = int(round(N))

    a1 = nu1 / 2.0
    b1 = a1

    a2 = nu2 / 2.0
    b2 = a2

    # Dimensions.
    d = 2

    Gs = np.array([Gs0, Gs1])

    # Gaussian correlation matrix.
    R = np.array(
        [[1.0, pcorr],
         [pcorr, 1.0]])

    R *= cov

    rv1 = invgamma(a1, scale=b1)
    rv2 = invgamma(a2, scale=b2)

    Zs = mvn(np.full(d, mean), R, size=N)

    Ws = np.empty(Zs.shape)

    Ss = np_rand(size=N)

    Ws[:, 0] = rv1.ppf(Ss)
    Ws[:, 1] = rv2.ppf(Ss)

    Xs = ((Ws ** 0.5) * Zs) + (Gs * Ws)

    probs = rankdata(Xs, axis=0) / (N + 1.0)

    return probs


def get_sid_tcop_probs_auto(args):

    pcorr, mean, cov, nu1, nu2, N, Gs1, Gs2 = args

    # Due to optimization.
    N = int(round(N))

    a1 = nu1 / 2.0
    b1 = a1

    a2 = nu2 / 2.0
    b2 = a2

    Gs = np.array([Gs1, Gs2])

    rv1 = invgamma(a1, scale=b1)
    rv2 = invgamma(a2, scale=b2)

    Ss = np_rand(size=N)

    if False:
        # Not correlated to Ws.
        Zs = np_norm(loc=0.0, scale=cov ** 0.5, size=N)

    else:
        # Correlated to Ws.
        Zs = norm.ppf(Ss, loc=0.0, scale=cov ** 0.5)

    for i in range(1, N):
        Zs[i] += Zs[i - 1] * pcorr

    Zs += mean

    Ws = np.empty((N, 2))

    Ws[:, 0] = rv1.ppf(1 - Ss)
    Ws[:, 1] = rv2.ppf(1 - Ss)
    for i in range(1, N):
        Ws[i,:] += Ws[i - 1,:] * pcorr

    Xs = ((Ws ** 0.5).sum(axis=1) * Zs) + (Gs * Ws).sum(axis=1)
    # Xs = ((np.sign(Gs) * (Ws ** 0.5)).sum(axis=1) * Zs) + (Gs * Ws).sum(axis=1)
    # Xs = ((Ws.sum(axis=1) ** 0.5) * Zs) + (Gs * Ws).sum(axis=1)
    # Xs = Zs + (Gs * Ws).sum(axis=1)
    # Xs = ((Ws ** 0.5).sum(axis=1)) + (Gs * Ws).sum(axis=1)
    # Xs = (((Ws ** 2).sum(axis=1) ** 0.5) * Zs) + (Gs * Ws).sum(axis=1)

    # Xs = Ws.sum(axis=1) + Zs

    # Xs = lag_var(Xs.reshape(-1, 1), 1)

    probs = rankdata(Xs, axis=0) / (N + 1.0)

    probs = probs.reshape(-1, 1)

    return probs


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    # Degrees of Freedom.
    # Apparently, for dir asymm.
    nu1 = 2.07
    nu2 = 3.07

    # Sample size.
    N = int(1e5)

    # Pearson correlation.
    pcorr = 0.97

    mean = 100.0

    # Scale pcorr matrix with this.
    cov = 10.0

    # Skewness constants.
    # Apparently, for order asymm.
    Gs = np.array([-1.0, -1.0, ])

    # Copula type
    # cop_type = 'cross'
    cop_type = 'auto'
    #==========================================================================

    if cop_type == 'cross':
        probs = get_sid_tcop_probs_cross((pcorr, mean, cov, nu1, nu2, N, *Gs))

    elif cop_type == 'auto':
        probs = get_sid_tcop_probs_auto((pcorr, mean, cov, nu1, nu2, N, *Gs))

    else:
        raise ValueError(f'Unknown cop_type: {cop_type}!')

    plt.scatter(probs[:, 0], probs[:, 1], alpha=10 / 255.0)

    plt.title(
        f'Copula type: {cop_type}\n'
        f'$\\nu$s: {nu1, nu2}, N: {N}, $\\gamma$s: ({Gs[0]}, {Gs[1]}), '
        f'pcorr: {pcorr}')

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.gca().set_aspect('equal')
    plt.show()

    plt.close()
    #==========================================================================

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
