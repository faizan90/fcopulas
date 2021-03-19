'''
@author: Faizan-Uni-Stuttgart

Jan 13, 2021

11:23:03 AM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
from mayavi import mlab
import matplotlib.pyplot as plt
from scipy.stats import rankdata

DEBUG_FLAG = False


def get_distance_from_vector(points, beg_vector, end_vector):

    assert points.ndim == 2
    assert beg_vector.ndim == 1
    assert beg_vector.size == end_vector.size
    assert points.shape[1] == beg_vector.size

    v = end_vector - beg_vector
    w = points - beg_vector

    # Relative distance on v, starting from beg_vector where w intersects v.
    t = (np.dot(w, v) / np.dot(v, v)).reshape(-1, 1)

    print('points:')
    print(points, end='\n\n')

    # The intersection point on the line p0-p1.
    c = t * v

    print('c:')
    print(c, end='\n\n')

    # Coordinates of pt w.r.t. the intersection point on the line p0-p1.
    d = w - c

    print('d:')
    print(d, end='\n\n')

    # Length of the vector.
#     print((d ** 2).sum(axis=1) ** 0.5)

    assert np.all(c >= 0) and np.all(c <= 1), 'Out of bounds!'
    return


def get_asymms_nd(data):

    assert data.ndim == 2

    n_pts, n_dims = data.shape

    if False:
        # Order asymmetry stuff.
        order_asm_beg_corners = np.zeros((n_dims, n_dims), dtype=bool)

        np.fill_diagonal(order_asm_beg_corners, 1)

        order_asm_end_corners = ~order_asm_beg_corners

        order_asm_beg_corners = order_asm_beg_corners.astype(int)
        order_asm_end_corners = order_asm_end_corners.astype(int)

    #     print(order_asm_beg_corners)
    #     print(order_asm_end_corners)

#         data = np.array([[1.0, 0.5, 0.5], [0.5, 0.5, 0.49], [0.5, 0.5, 0.0]])

        get_distance_from_vector(
            data,
            order_asm_beg_corners[0,:],
            order_asm_end_corners[0,:])

    if True:
        # Directional asymmetry stuff.
        dir_asm_beg_corner = np.zeros(n_dims, dtype=int)
        dir_asm_end_corner = np.ones(n_dims, dtype=int)

        get_distance_from_vector(
            data,
            dir_asm_beg_corner,
            dir_asm_end_corner)

    return


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\Copulas_practice\asymms_nd')
    os.chdir(main_dir)

    in_data_file = Path(
        r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    in_df = pd.read_csv(in_data_file, sep=';', index_col=0)

    in_df = in_df[['420', '427']]

    in_df = in_df.loc['1961-01-01':'1961-12-31']

    assert np.all(np.isfinite(in_df.values))

    # Station cmpr.
    in_data = in_df.values

    in_probs = np.apply_along_axis(
        rankdata, 0, in_data) / (in_data.shape[0] + 1.0)

    get_asymms_nd(in_probs[::30,:])
    return


if __name__ == '__main__':

    _save_log_ = False
    if _save_log_:
        from datetime import datetime
        from std_logger import StdFileLoggerCtrl

        # save all console activity to out_log_file
        out_log_file = os.path.join(
            r'P:\Synchronize\python_script_logs\\%s_log_%s.log' % (
            os.path.basename(__file__),
            datetime.now().strftime('%Y%m%d%H%M%S')))

        log_link = StdFileLoggerCtrl(out_log_file)

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
            import pdb
            pdb.post_mortem()

    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
