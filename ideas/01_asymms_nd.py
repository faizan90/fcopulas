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
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Even though unsed, this has to be imported.
from scipy.stats import rankdata
DEBUG_FLAG = False


def determinant(M):

    # Laplace's expansion from Wikipedia.

    # Base case of recursive function: 2x2 matrix (such that det(M) = ad - cb)
    if len(M) == 2:
        return (M[0][0] * M[1][1]) - (M[0][1] * M[1][0])

    else:

        total = 0
        for column, element in enumerate(M[0]):
            # Exclude first row and current column.
            K = [x[:column] + x[column + 1:] for x in M[1:]]

            # Given that the element is in row 1, sign is negative if index is odd.
            if column % 2 == 0:
                total += element * determinant(K)

            else:
                total -= element * determinant(K)

        return total


def get_normal_vector(M):

    # First row is ones.

    norm_vec = len(M[0]) * [0]

    for column, element in enumerate(M[0]):
        K = [x[:column] + x[column + 1:] for x in M[1:]]

        if column % 2:
            sgn = -1.0

        else:
            sgn = +1.0

        norm_vec[column] = sgn * element * determinant(K)

    return norm_vec


def normal_vector_stuff():

    # p0 = np.array([0.0, 0.0, 0.0])
    # p1 = np.array([1.0, 1.0, 1.0])
    # pt = np.array([1.0, 0.0, 1.0])

    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 1.0])
    pt = np.array([1.0, 0.0])

    v = p1 - p0
    w = pt - p0

    t = np.dot(w, v) / np.dot(v, v)

    d = w - (t * v)

    print(d)
    print((d ** 2).sum() ** 0.5)

    # print(np.cross(np.array([0, 0, 1, 1]), np.array([1, 1, 1, 1])))

    print((np.array([-2, 2, -1]) * np.array([0, 1, 2])).sum())

    M = [
        [1, 1, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 1, 1],
        [1, 1, 1, 1],
        ]

    norm_vec = get_normal_vector(M)

    print(norm_vec)

    M = np.array(M)
    norm_vec = np.array(norm_vec)

    print((norm_vec * M[1,:]).sum())
    print((norm_vec * M[2,:]).sum())
    print((norm_vec * M[3,:]).sum())
    print((norm_vec * np.array([0, 3, 0, 0])).sum())

    return


def get_mag_and_phs_spec(data):

    assert data.ndim == 2

    ft = np.fft.rfft(data, axis=0)

    mag_spec = np.abs(ft)
    phs_spec = np.angle(ft)

    return mag_spec, phs_spec


def get_phs_rand_sers(data):

    assert data.ndim == 2

    mag_spec, phs_spec = get_mag_and_phs_spec(data)

    phs_spec_rand = phs_spec.copy()

    x = (-np.pi + (
        2 * np.pi * np.random.random(phs_spec.shape[0] - 2))).reshape(-1, 1)

    phs_spec_rand[1:-1,:] += x

    ft = np.empty_like(mag_spec, dtype=complex)

    ft.real = mag_spec * np.cos(phs_spec_rand)
    ft.imag = mag_spec * np.sin(phs_spec_rand)

    out_data = np.fft.irfft(ft, axis=0)

    return out_data


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

        for i in range(fin_lag):
            out_arr.append(in_arr[i:-(fin_lag - i)])

        out_arr = np.concatenate(out_arr, axis=1)

    return out_arr


def unit_cube_stuff(probs):

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

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

    ax.plot(
        ucube_pts[:, 0],
        ucube_pts[:, 1],
        ucube_pts[:, 2],
        label='Unit Cube')

    ax.scatter(probs[:, 0], probs[:, 1], probs[:, 2], alpha=0.3, c='k')

    diag_x = np.linspace(0, 1, 10)
    diag_y = np.linspace(0, 1, 10)

    diag_x, diag_y = np.meshgrid(diag_x, diag_y)

    diag_z = diag_x

    ax.plot_surface(
        diag_y, diag_y, diag_z, linewidth=0, antialiased=False, alpha=0.7)

    print(np.round(diag_x, 2))
    print(np.round(diag_y, 2))
    print(np.round(diag_z, 2))

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.legend()

    plt.show()

    return


def unit_cube_stuff_mayavi(probs, title, ax_labels):

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
        opacity=0.3,
        color=(0, 0, 0),
        scale_factor=0.05)

#     diag_x = np.linspace(0, 1, 10)
#     diag_y = np.linspace(0, 1, 10)
#
#     diag_x, diag_y = np.meshgrid(diag_x, diag_y)

    # Independent of z.
#     diag_z = diag_x
#
#     mlab.mesh(
#         diag_y,
#         diag_y,
#         diag_z,
#         opacity=0.7,
#         color=(245 / 255, 135 / 255, 66 / 255))

    # Independent of x.
#     diag_z = diag_y
#
#     mlab.mesh(
#         diag_x,
#         diag_y,
#         diag_z,
#         opacity=0.7,
#         color=(245 / 255, 135 / 255, 66 / 255))
#

#     print(np.round(diag_x, 2))
#     print(np.round(diag_y, 2))
#     print(np.round(diag_z, 2))
#
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
    return


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\Copulas_practice\asymms_nd')
    os.chdir(main_dir)

    in_data_file = Path(
        r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    in_df = pd.read_csv(in_data_file, sep=';', index_col=0)

    in_df = in_df[['420', '427', '3421']]

    in_df = in_df.loc['1961-01-01':'1980-12-31']

    assert np.all(np.isfinite(in_df.values))

    # Station cmpr.
    in_data = in_df.values

    in_probs = np.apply_along_axis(
        rankdata, 0, in_data) / (in_data.shape[0] + 1.0)

    norm_data = get_phs_rand_sers(in_data)

    norm_probs = np.apply_along_axis(
        rankdata, 0, norm_data) / (norm_data.shape[0] + 1.0)

    print(np.corrcoef(in_data, rowvar=False))
    print(np.corrcoef(norm_data, rowvar=False))

#     unit_cube_stuff(in_probs)
    unit_cube_stuff_mayavi(in_probs, 'observed', in_df.columns)
    unit_cube_stuff_mayavi(norm_probs, 'normal', in_df.columns)

    # Lag cmpr.
#     in_data = lag_var(in_df.values[:, 0].reshape(-1, 1), 3)
#
#     in_probs = np.apply_along_axis(
#         rankdata, 0, in_data) / (in_data.shape[0] + 1.0)
#
#     norm_data = get_phs_rand_sers(in_data)
#
#     norm_probs = np.apply_along_axis(
#         rankdata, 0, norm_data) / (norm_data.shape[0] + 1.0)
#
#     print(np.corrcoef(in_data, rowvar=False))
#     print(np.corrcoef(norm_data, rowvar=False))
#
#     unit_cube_stuff_mayavi(in_probs, 'observed', [0, 1, 2])
#     unit_cube_stuff_mayavi(norm_probs, 'normal', [0, 1, 2])

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
