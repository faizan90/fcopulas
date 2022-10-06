'''
@author: Faizan-Uni-Stuttgart

Oct 5, 2021

5:22:18 PM

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

from fcopulas import get_distances_from_vector_nd, get_asymms_nd_v2_raw_cy

# Matplotlib error.
# from aa_asymms_nd import get_distance_from_vector

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\Copulas_practice\asymms_nd\mvn_discharge_auto_420_5_sims')

    os.chdir(main_dir)

    fig_size = (10, 7)
    dpi = 150
    alpha = 0.4
    #==========================================================================

    obs_file = Path('obs.csv')

    asymms_obs = get_asymms(obs_file)

    asymms_sims = []

    for sim_file in main_dir.glob('./sim_*.csv'):
        asymms_sims.append(get_asymms(sim_file))

    n_asymms = len(asymms_obs)

    x_crds = np.arange(n_asymms - 1)

    for asymm_idx in range(n_asymms):

        print('Asymm_idx:', asymm_idx)

        plt.figure(figsize=fig_size)

        leg_flag = True
        for asymms_sim in asymms_sims:

            if leg_flag:
                label = 'GAU'
                leg_flag = False

            else:
                label = None

            plt.plot(
                x_crds,
                asymms_sim[asymm_idx],
                alpha=alpha,
                c='b',
                label=label)

        plt.plot(
            x_crds, asymms_obs[asymm_idx], alpha=0.9, c='r', label='OBS')

        plt.xlabel('Asymmetry component index (-)')
        plt.ylabel('Component (-)')

        plt.title(f'{asymm_idx}th asymmetry, nD: {n_asymms - 1}')

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.legend()

        plt.gca().xaxis.get_major_locator().set_params(integer=True)

        plt.savefig(
            f'test_asymms_cmpr_{asymm_idx}.png', bbox_inches='tight', dpi=dpi)

        plt.close()

    return


def get_distance_from_vector(points, beg_vector, end_vector):

    vb = False

    assert points.ndim == 2
    assert beg_vector.ndim == 1
    assert beg_vector.size == end_vector.size
    assert points.shape[1] == beg_vector.size

    v = end_vector - beg_vector
    w = points - beg_vector

    # Relative distance on v, starting from beg_vector where w intersects v.
    t = (np.dot(w, v) / np.dot(v, v)).reshape(-1, 1)

    assert np.all(t >= 0) and np.all(t <= 1), 'Out of bounds!'

    if vb:
        print('points:')
        print(points, end='\n\n')

    # The intersection point on the line p0-p1.
    c = (t * v) + beg_vector

    assert np.all(c >= 0) and np.all(c <= 1), 'Out of bounds!'

    if vb:
        print('c:')
        print(c, end='\n\n')

    # Coordinates of pt w.r.t. the intersection point on the line v.
    d = points - c

    if vb:
        print('d:')
        print(d, end='\n\n')

    # Unit vectors of d.
#     du = d / ((d ** 2).sum(axis=1) ** 0.5).reshape(-1, 1)

    # Length of the vector.
#     print((d ** 2).sum(axis=1) ** 0.5)

    # return np.round(d, 15), np.round(c, 15)
    return d, c


def get_asymms_nd_v2(data, norm_flag):

    assert np.all(data >= 0) and np.all(data <= 1)

    assert data.ndim == 2

    n_dims = data.shape[1]

    out_asymms = []

    # Order asymmetries.
    order_asm_end_corners = np.zeros((n_dims, n_dims), dtype=bool)

    np.fill_diagonal(order_asm_end_corners, 1)

    order_asm_beg_corners = ~order_asm_end_corners

    order_asm_beg_corners = order_asm_beg_corners.astype(np.float64)
    order_asm_end_corners = order_asm_end_corners.astype(np.float64)

    for i in range(n_dims):
        # distances, _ = get_distance_from_vector(
        #     data,
        #     order_asm_beg_corners[i,:],
        #     order_asm_end_corners[i,:])

        distances = get_distances_from_vector_nd(
            data.copy(order='c'),
            order_asm_beg_corners[i,:],
            order_asm_end_corners[i,:])

        # assert np.all(np.isclose(distances, distances_cy))

        dist_unit = (distances ** 3).mean(axis=0)

        if norm_flag:
            dist_unit /= ((dist_unit ** 2).sum() ** 0.5)

        out_asymms.append(dist_unit)

        if False:
            dist_unit_dist = ((dist_unit ** 2).sum()) ** 0.5

            print(
                f'AO({i}):',
                order_asm_beg_corners[i,:],
                dist_unit,
                round(dist_unit_dist, 4))

    # Directional asymmetry.
    dir_asm_beg_corner = np.zeros(n_dims, dtype=int)
    dir_asm_end_corner = np.ones(n_dims, dtype=int)

    distances, _ = get_distance_from_vector(
        data,
        dir_asm_beg_corner,
        dir_asm_end_corner)

    dist_unit = (distances ** 3).mean(axis=0)

    if norm_flag:
        dist_unit /= ((dist_unit ** 2).sum() ** 0.5)

    out_asymms.append(dist_unit)

    if False:
        dist_unit_dist = ((dist_unit ** 2).sum()) ** 0.5

        print(
            'AD   :',
            dir_asm_beg_corner,
            dist_unit,
            round(dist_unit_dist, 4))

    out_asymms_test = np.array(out_asymms)
    assert np.all(np.isclose(get_asymms_nd_v2_raw_cy(data.copy(order='c')), out_asymms_test))

    return out_asymms


def get_asymms(in_file):

    in_df = pd.read_csv(in_file, sep=';', index_col=0)

    assert np.all(np.isfinite(in_df.values))

    in_data = in_df.values

    in_probs = np.apply_along_axis(
        rankdata, 0, in_data) / (in_data.shape[0] + 1.0)

    # return get_asymms_nd_v2(in_probs, False)
    return get_asymms_nd_v2_raw_cy(in_probs.copy(order='c'))


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
