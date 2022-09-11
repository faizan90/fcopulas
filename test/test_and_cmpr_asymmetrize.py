'''
@author: Faizan-Uni-Stuttgart

Aug 10, 2022

3:13:49 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from fcopulas import asymmetrize_type_1_cy

DEBUG_FLAG = False


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    n_time_iters = 50

    n_steps = 10000
    n_cols = 2

    n_levels = 1000
    max_shift_exp = 0.5
    max_shift = 5
    pre_vals_ratio = 0.95
    asymms_rand_err = -1e-6 + (2e-6) * np.random.random(n_steps)

    data = 10 + (20) * np.random.random(size=(n_steps, n_cols))

    probs = rankdata(data, axis=0) / (n_steps + 1.0)

    data = data.copy(order='f')
    probs = probs.copy(order='f')
    asymms_rand_err = asymms_rand_err.copy(order='f')
    #==========================================================================

    print('python')
    beg_time = timeit.default_timer()

    for _ in range(n_time_iters):
        asymm_data_py = asymmetrize_data_type_1_py(
            data,
            probs,
            n_levels,
            max_shift_exp,
            max_shift,
            pre_vals_ratio,
            asymms_rand_err,
            )

    python_time = timeit.default_timer() - beg_time

    print(f'Python time for {n_time_iters}: {python_time:0.3f}')

    print('cython')
    beg_time = timeit.default_timer()

    for _ in range(n_time_iters):
        asymm_data_cy = asymmetrize_type_1_cy(
            data,
            probs,
            n_levels,
            max_shift_exp,
            max_shift,
            pre_vals_ratio,
            1,
            asymms_rand_err,
            )

    cython_time = timeit.default_timer() - beg_time

    print(f'Cython time for {n_time_iters}: {cython_time:0.3f}')

    print(
        f'Cython is {(python_time - cython_time) / cython_time:0.2f} '
        f'times faster!')

    # x = np.sort(asymm_data_py, axis=0)
    # y = np.sort(asymm_data_cy, axis=0)

    # x = asymm_data_py
    # y = asymm_data_cy
    #
    # for i in range(n_steps):
    #     print(x[i], y[i])

    assert np.all(np.isclose(asymm_data_py, asymm_data_cy))
    return


def asymmetrize_data_type_1_py(
        data,
        probs,
        n_levels,
        max_shift_exp,
        max_shift,
        pre_vals_ratio,
        asymms_rand_err,
        ):

    out_data = np.empty_like(data)

    for i in range(data.shape[1]):

        data_i = data[:, i].copy()
        probs_i = probs[:, i].copy()

        vals_sort = np.sort(data_i)

        levels = (probs_i * n_levels).astype(int)

        asymm_vals = data_i.copy()
        for level in range(n_levels):
            asymm_vals_i = asymm_vals.copy()

            max_shift_level = (
                max_shift -
                (max_shift * ((level / n_levels) ** max_shift_exp)))

            max_shift_level = int(max_shift_level)

            if max_shift_level == 0:
                break

            elif max_shift_level < 0:
                shifts = range(max_shift_level, 0, 1)

            else:
                shifts = range(1, max_shift_level + 1)

            # print(asymm_vals_i)
            for shift in shifts:
                asymm_vals_i = np.roll(asymm_vals_i, shift)

                # print(level, max_shift_level, asymm_vals_i)

                idxs_to_shift = levels <= level

                asymm_vals[idxs_to_shift] = (
                    asymm_vals[idxs_to_shift] * pre_vals_ratio +
                    asymm_vals_i[idxs_to_shift] * (1.0 - pre_vals_ratio))

        # if asymms_rand_err is not None:
        asymm_vals += asymms_rand_err

        asymm_vals = vals_sort[np.argsort(np.argsort(asymm_vals))]

        out_data[:, i] = asymm_vals

    return out_data


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
