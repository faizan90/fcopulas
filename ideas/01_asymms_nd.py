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
# import matplotlib.pyplot as plt
from scipy.stats import rankdata
from mpl_toolkits.mplot3d import Axes3D  # Even though unsed, this has to be imported.

DEBUG_FLAG = False


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

    return d, c


def get_asymms_nd(data):

    assert data.ndim == 2

    n_pts, n_dims = data.shape

    asymm_dir_pts = np.array([
        [1, 0, 0],
        [0, 1, 1]
        ], dtype=np.float64)

    if True:
        # Order asymmetry stuff.
        order_asm_end_corners = np.zeros((n_dims, n_dims), dtype=bool)

        np.fill_diagonal(order_asm_end_corners, 1)

        order_asm_beg_corners = ~order_asm_end_corners

        order_asm_beg_corners = order_asm_beg_corners.astype(int)
        order_asm_end_corners = order_asm_end_corners.astype(int)

    #     print(order_asm_beg_corners)
    #     print(order_asm_end_corners)

#         data = np.array([[1.0, 0.5, 0.5], [0.5, 0.5, 0.49], [0.5, 0.5, 0.0]])

        for i in range(n_dims):
            distances, c = get_distance_from_vector(
                data,
                order_asm_beg_corners[i,:],
                order_asm_end_corners[i,:])

            dist_unit = (distances ** 3).mean(axis=0)
            dist_unit /= ((dist_unit ** 2).sum() ** 0.5)

            print(
                f'AO({i}):',
                order_asm_beg_corners[i,:], dist_unit,)

            if n_dims == 3:
                asymm_dir_pts[0,:] = order_asm_beg_corners[i,:]
                asymm_dir_pts[1,:] = order_asm_end_corners[i,:]

                unit_cube_stuff_mayavi(
                    data, 'test', ['x', 'y', 'z'], c, asymm_dir_pts)

    if True:
        # Directional asymmetry stuff.
        dir_asm_beg_corner = np.zeros(n_dims, dtype=int)
        dir_asm_end_corner = np.ones(n_dims, dtype=int)

        distances, c = get_distance_from_vector(
            data,
            dir_asm_beg_corner,
            dir_asm_end_corner)

        dist_unit = (distances ** 3).mean(axis=0)
#         dist_unit /= ((dist_unit ** 2).sum() ** 0.5)

        print('AD   :', dir_asm_beg_corner, dist_unit)

        if n_dims == 3:
            asymm_dir_pts[0,:] = dir_asm_beg_corner
            asymm_dir_pts[1,:] = dir_asm_end_corner

#             unit_cube_stuff_mayavi(
#                 data, 'test', ['x', 'y', 'z'], c, asymm_dir_pts)

    return


def unit_cube_stuff_mayavi(probs, title, ax_labels, c, asymm_dir_pts):

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

    if asymm_dir_pts is not None:
        mlab.plot3d(
            asymm_dir_pts[:, 0],
            asymm_dir_pts[:, 1],
            asymm_dir_pts[:, 2],
    #         color=(0.5, 0.5, 0.5),
            )

    if c is None:
        mlab.points3d(
            probs[:, 0],
            probs[:, 1],
            probs[:, 2],
            opacity=0.3,
            color=(0, 0, 0),
            scale_factor=0.05)

    else:
        for i in range(c.shape[0]):
            line_pts = np.empty((2, 3), dtype=np.float64)

            line_pts[0,:] = c[i,:]
            line_pts[1,:] = probs[i,:]

            mlab.plot3d(
                line_pts[:, 0],
                line_pts[:, 1],
                line_pts[:, 2],
                )

        mlab.points3d(
            probs[:, 0],
            probs[:, 1],
            probs[:, 2],
            opacity=0.3,
            color=(0, 0, 0),
            scale_factor=0.05)

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

    mlab.show(stop=True)
#     mlab.close()
    return


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\Copulas_practice\asymms_nd')
    os.chdir(main_dir)

    if False:
        get_asymms_nd(np.repeat(np.linspace(0, 1, 1000), 3).reshape(-1, 3))

    else:
        in_data_file = Path(
            r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

        in_df = pd.read_csv(in_data_file, sep=';', index_col=0)

        in_df = in_df[['2446', '1462', '4427']]

        in_df = in_df.loc['1990-01-01':'1990-12-31']

        assert np.all(np.isfinite(in_df.values))

        # Station cmpr.
        in_data = in_df.values

        in_probs = np.apply_along_axis(
            rankdata, 0, in_data) / (in_data.shape[0] + 1.0)

        get_asymms_nd(in_probs)

    #     unit_cube_stuff_mayavi(in_probs, 'observed', in_df.columns)
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
