'''
@author: Faizan-Uni-Stuttgart

Nov 18, 2021

3:33:17 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
from mayavi import mlab
from scipy.stats import rankdata
import matplotlib.pyplot as plt; plt.ioff()
from mpl_toolkits.mplot3d import Axes3D  # Even though unsed, this has to be imported.

from fcopulas import (
    get_srho_plus_for_probs_nd,
    get_srho_minus_for_probs_nd,
    )

from amb_cmpr_mvn_asymms_to_obs import get_asymms_nd_v2

DEBUG_FLAG = True


def unit_cube_stuff_mayavi(probs, title, ax_labels):

    assert probs.shape[1] == 3

    mlab.figure()

    # x, y, z
    ucube_pts = np.array([
        [0, 0, 0],  # lower square
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 0],

        [0, 0, 1],  # top square
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 1],

        [1, 0, 1],
        [1, 0, 0],

        [1, 1, 0],
        [1, 1, 1],

        [0, 1, 1],
        [0, 1, 0],
        ], dtype=np.float64)

    mlab.plot3d(
        ucube_pts[:, 0],
        ucube_pts[:, 1],
        ucube_pts[:, 2])

    mlab.points3d(
        probs[:, 0],
        probs[:, 1],
        probs[:, 2],
        opacity=10 / 255.0,
        color=(0, 0, 0),
        scale_factor=0.1)

#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')

    mlab.axes(
        extent=(0, 1, 0, 1, 0, 1),
        xlabel=ax_labels[0],
        ylabel=ax_labels[1],
        zlabel=ax_labels[2],
        nb_labels=5)

    mlab.title(title)

    print('Mayavi plot active...')
    mlab.show(stop=True)
#     mlab.close()
    return


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\Copulas_practice\asymms_max')
    os.chdir(main_dir)

    # Not fully correct.

    n_dims = 3

    if n_dims == 2:
        n_vals = int(3e3)

    elif n_dims == 3:
        n_vals = int(3e3)  # + 1

    elif n_dims == 4:
        n_vals = int(4e3)

    else:
        raise ValueError(n_dims)

    out_dir = Path(f'{n_dims}D_0y')

    fig_size = (10, 7)
    dpi = 150
    alpha = 0.6
    #==========================================================================

    assert not (n_vals % n_dims), n_vals % n_dims

    assert n_dims >= 2

    out_dir.mkdir(exist_ok=True)

    probs = np.tile(
        np.arange(1, n_vals + 1).reshape(-1, 1),
        (1, n_dims)) / (n_vals + 1.0)

    print('Computing asymms...')
    asymms_alpha = [get_asymms_nd_v2(probs, False)]

    scorrs = [
        [get_srho_plus_for_probs_nd(probs),
         get_srho_minus_for_probs_nd(probs)]]

    alpha_ps = [0.0]

    for i in range(n_dims, n_vals + n_dims - 1, n_dims):

        # Don't use +1 with n_vals in denominator.
        alpha_p = (i) / (n_vals)
        probs_alpha = probs.copy()

        flip_idxs = probs_alpha[:, 1] <= alpha_p

        n_flip_idxs = flip_idxs.sum()

        assert not (n_flip_idxs % n_dims), (i, n_flip_idxs, n_dims)
        assert n_flip_idxs == i

        probs_alpha[flip_idxs,:] = alpha_p / n_dims

        sclrs = np.repeat(np.linspace(1.0, 0, n_flip_idxs // n_dims), n_dims)

        for j, sclr in zip(range(n_flip_idxs), sclrs):

            for k in range(n_dims):
                if k == (j % n_dims):
                    probs_alpha[j, k] += (
                        alpha_p * ((n_dims - 1) / n_dims)) * sclr

                else:
                    probs_alpha[j, k] -= (alpha_p / n_dims) * sclr

            if False:
                # Uniform spread on a hypersurface. I am not sure. Looks so.
                # The final asymmetry value depends if the spread is used
                # or not.
                rand_k_vec = np.random.random(n_dims)
                rand_k_vec *= probs_alpha[j,:] ** 0.5

                rand_k_sum = rand_k_vec.sum()

                prob_rand_k_vec = alpha_p * (rand_k_vec / rand_k_sum)

                probs_alpha[j,:] = prob_rand_k_vec

        assert np.all(
            np.isclose(probs_alpha[flip_idxs,:].sum(axis=1), alpha_p))

        # Without converting to ranks, the probs_alpha have non-uniform
        # values. This creates a problem for the Spearman correlation as
        # it requires the inputs to be uniformly distributed.
        probs_ranks = (rankdata(probs_alpha, method='min', axis=0))
        probs_alpha = (probs_ranks / (probs_ranks.max() + 1.0)).copy(order='c')

        # if i == n_vals:
        #     if n_dims == 3:
        #         plt.hist(probs_alpha[:, 0], rwidth=0.9, alpha=0.7)
        #         plt.hist(probs_alpha[:, 1], rwidth=0.8, alpha=0.7)
        #         plt.hist(probs_alpha[:, 2], rwidth=0.7, alpha=0.7)
        #         plt.show(block=False)
        #
        #         unit_cube_stuff_mayavi(probs_alpha, 'test', ['x', 'y', 'z'])
        #
        #     elif n_dims == 2:
        #         plt.scatter(probs_alpha[:, 0], probs_alpha[:, 1], alpha=0.5)
        #         plt.gca().set_aspect('equal')
        #         plt.grid()
        #         plt.gca().set_axisbelow(True)
        #         plt.show()
        #         plt.close()
        #
        #     elif n_dims == 4:
        #         plt.hist(probs_alpha[:, 0], rwidth=0.9, alpha=0.7)
        #         plt.hist(probs_alpha[:, 1], rwidth=0.8, alpha=0.7)
        #         plt.hist(probs_alpha[:, 2], rwidth=0.7, alpha=0.7)
        #         plt.hist(probs_alpha[:, 3], rwidth=0.6, alpha=0.7)
        #
        #     else:
        #         pass

        asymms_alpha.append(get_asymms_nd_v2(probs_alpha, False))

        scorrs.append(
            [get_srho_plus_for_probs_nd(probs_alpha),
             get_srho_minus_for_probs_nd(probs_alpha)])

        alpha_ps.append(alpha_p)

    asymms_alpha = np.array(asymms_alpha)

    scorrs = np.array(scorrs)

    alpha_ps = np.array(alpha_ps)

    print(f'n_vals: {n_vals}')
    print(f'n_alpha_ps: {alpha_ps.size}')
    print(f'Scorrs min., max.: {scorrs.min()}, {scorrs.max()}')

    print('Plotting figures...')
    plt.figure(figsize=fig_size)
    for asymm_idx in range(asymms_alpha.shape[1]):

        # Alphas vs. asymms.
        plt.title(f'asymm_idx: {asymm_idx}')
        for comp_idx in range(asymms_alpha.shape[2]):
            asymm_comp_vals = asymms_alpha[:, asymm_idx, comp_idx]

            plt.plot(
                alpha_ps, asymm_comp_vals, label=f'ci: {comp_idx}', alpha=alpha)

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.xlabel('Alpha (-)')
        plt.ylabel('Asymmetry component value (-)')

        plt.legend()

        # plt.show()

        plt.savefig(
            out_dir / f'alphas_asymms_{asymm_idx}.png',
            dpi=dpi,
            bbox_inches='tight')

        plt.clf()

        # Scorr (+ve) vs. asymms.
        plt.title(f'asymm_idx: {asymm_idx}')
        for comp_idx in range(asymms_alpha.shape[2]):
            asymm_comp_vals = asymms_alpha[:, asymm_idx, comp_idx]

            plt.plot(
                scorrs[:, 0],
                asymm_comp_vals,
                label=f'ci: {comp_idx}',
                alpha=alpha)

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.xlabel('Spearman correlation +ve (-)')
        plt.ylabel('Asymmetry component value (-)')

        plt.legend()

        # plt.show()

        plt.savefig(
            out_dir / f'scorrpve__asymms_{asymm_idx}.png',
            dpi=dpi,
            bbox_inches='tight')

        plt.clf()

        # Scorr (-ve) vs. asymms.
        plt.title(f'asymm_idx: {asymm_idx}')
        for comp_idx in range(asymms_alpha.shape[2]):
            asymm_comp_vals = asymms_alpha[:, asymm_idx, comp_idx]

            plt.plot(
                scorrs[:, 1],
                asymm_comp_vals,
                label=f'ci: {comp_idx}',
                alpha=alpha)

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.xlabel('Spearman correlation -ve (-)')
        plt.ylabel('Asymmetry component value (-)')

        plt.legend()

        # plt.show()

        plt.savefig(
            out_dir / f'scorrnve__asymms_{asymm_idx}.png',
            dpi=dpi,
            bbox_inches='tight')

        plt.clf()

    plt.plot(alpha_ps, scorrs[:, 0], label='+ve', alpha=alpha)
    plt.plot(alpha_ps, scorrs[:, 1], label='-ve', alpha=alpha)

    plt.plot(
        alpha_ps, scorrs[:, 0] - scorrs[:, 1], label='+ve - -ve', alpha=alpha)

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.xlabel('Alpha (-)')
    plt.ylabel('Spearman correlation (-)')

    plt.legend()

    # plt.show()

    plt.savefig(
        out_dir / f'alphas_scorrs.png',
        dpi=dpi,
        bbox_inches='tight')

    plt.clf()

    plt.close()
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
