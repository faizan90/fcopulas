'''
@author: Faizan-Uni-Stuttgart

Apr 15, 2021

5:12:59 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\Copulas_practice\mv_gaussian_dependence_test')

    os.chdir(main_dir)

    n_dims = 3

    sample_size = int(1e4)

    covs = np.array([
        [1, 0.84, 0.89],
        [0.84, 1, 0.87],
        [0.89, 0.87, 1]
        ], dtype=np.float32)

    assert covs.shape[0] == n_dims
    assert covs.shape[1] == n_dims

    assert n_dims == 3, 'Meant for 3D only!'

    gau_vecs = np.random.multivariate_normal(
        np.zeros(n_dims), covs, size=sample_size)

    print('Gaussian sample:')
    print(gau_vecs)

    print('Gaussian sample covs:')
    print(np.corrcoef(gau_vecs, rowvar=False))

    gau_signs = np.sign(gau_vecs)

    gau_vecs_mod = gau_vecs.copy()

    opt_indics = np.full(sample_size, np.nan)

    for i in range(sample_size):
        if np.all(gau_signs[i] == -1):
            opt_indics[i] = 0

            gau_vecs_mod[i] *= -1

        elif np.prod(gau_signs) == -1:
            opt_indics[i] = 1

            gau_vecs_mod[i] *= -1

        else:
            opt_indics[i] = -1

    gau_probs = rankdata(gau_vecs, axis=0) / (sample_size + 1.0)

    gau_mod_probs = rankdata(gau_vecs_mod, axis=0) / (sample_size + 1.0)

    for comb in combinations(np.arange(n_dims), 2):
        axes = plt.subplots(
            1,
            2,
            squeeze=False,
            figsize=(10, 5),
            sharex=True,
            sharey=True)[1]

        # Gaussian.
        axes[0, 0].scatter(
            gau_probs[:, comb[0]],
            gau_probs[:, comb[1]],
            c='k',
            alpha=0.5)

        axes[0, 0].set_xlabel(comb[0])
        axes[0, 0].set_ylabel(comb[1])

        axes[0, 0].grid()
        axes[0, 0].set_axisbelow(True)

        # Gaussian modified.
        axes[0, 1].scatter(
            gau_mod_probs[:, comb[0]],
            gau_mod_probs[:, comb[1]],
            c='k',
            alpha=0.5)

        axes[0, 1].set_xlabel(comb[0])

        axes[0, 1].grid()
        axes[0, 1].set_axisbelow(True)

        plt.savefig(f'{comb[0]}_{comb[1]}.png', bbox_inches='tight')
        plt.close()

    # Gaussian in 3D.
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(
        gau_probs[:, 0],
        gau_probs[:, 1],
        gau_probs[:, 2],
        alpha=0.5,
        c='k')

    plt.grid()

#     plt.show()

    plt.savefig('gau_0_1_2.png', bbox_inches='tight')
    plt.close()

    # Gaussian modified in 3D.
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(
        gau_mod_probs[:, 0],
        gau_mod_probs[:, 1],
        gau_mod_probs[:, 2],
        alpha=0.5,
        c='k')

    plt.grid()

#     plt.show()

    plt.savefig('gau_0_1_2_mod.png', bbox_inches='tight')
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
