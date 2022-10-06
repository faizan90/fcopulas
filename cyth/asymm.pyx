# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

import numpy as np
cimport numpy as np

cdef float asymms_exp = 3.0

cpdef float get_asymms_exp():

    return asymms_exp


cpdef tuple get_asymms_sample(DT_D[:] u, DT_D[:] v) except +:

    cdef:
        Py_ssize_t i, n_vals
        DT_D asymm_1, asymm_2

    n_vals = u.shape[0]

    asymm_1 = 0.0
    asymm_2 = 0.0

    for i in range(n_vals):
        asymm_1 += (u[i] + v[i] - 1.0)**asymms_exp
        asymm_2 += (u[i] - v[i])**asymms_exp

    asymm_1 = asymm_1 / n_vals
    asymm_2 = asymm_2 / n_vals
    return asymm_1, asymm_2


cpdef DT_D get_asymm_1_sample(DT_D[:] u, DT_D[:] v) except +:

    cdef:
        Py_ssize_t i, n_vals
        DT_D asymm_1

    n_vals = u.shape[0]

    asymm_1 = 0.0
    for i in range(n_vals):
        asymm_1 += (u[i] + v[i] - 1.0)**asymms_exp

    asymm_1 = asymm_1 / n_vals
    return asymm_1


cpdef DT_D get_asymm_2_sample(DT_D[:] u, DT_D[:] v) except +:

    cdef:
        Py_ssize_t i, n_vals
        DT_D asymm_2

    n_vals = u.shape[0]

    asymm_2 = 0.0
    for i in range(n_vals):
        asymm_2 += (u[i] - v[i])**asymms_exp

    asymm_2 = asymm_2 / n_vals
    return asymm_2


cpdef DT_D get_asymm_1_max(DT_D scorr) except +:

    cdef:
        DT_D a_max

    a_max = (
        0.5 * (1 - scorr)) * (1 - ((0.5 * (1 - scorr)) ** (1.0 / 3.0)))

    return a_max


cpdef DT_D get_asymm_2_max(DT_D scorr) except +:

    cdef:
        DT_D a_max

    a_max = (
        0.5 * (1 + scorr)) * (1 - ((0.5 * (1 + scorr)) ** (1.0 / 3.0)))

    return a_max


cpdef DT_D get_asymm_1_var(DT_D[:] u, DT_D[:] v, DT_D asymm_1_mean) except +:

    cdef:
        Py_ssize_t i, n_vals
        DT_D asymm_1_var

    n_vals = u.shape[0]

    asymm_1_var = 0.0
    for i in range(n_vals):
        asymm_1_var += (((u[i] + v[i] - 1.0)**asymms_exp) - asymm_1_mean)**2

    asymm_1_var /= n_vals

    return asymm_1_var


cpdef DT_D get_asymm_1_skew(
        DT_D[:] u, DT_D[:] v, DT_D asymm_1_mean, DT_D asymm_1_var) except +:

    cdef:
        Py_ssize_t i, n_vals
        DT_D asymm_1_skew

    n_vals = u.shape[0]

    asymm_1_skew = 0.0
    for i in range(n_vals):
        asymm_1_skew += (((u[i] + v[i] - 1.0)**asymms_exp) - asymm_1_mean)**3

    asymm_1_skew /= n_vals

    asymm_1_skew /= asymm_1_var**1.5

    return asymm_1_skew


cpdef DT_D get_asymm_2_var(DT_D[:] u, DT_D[:] v, DT_D asymm_2_mean) except +:

    cdef:
        Py_ssize_t i, n_vals
        DT_D asymm_2_var

    n_vals = u.shape[0]

    asymm_2_var = 0.0
    for i in range(n_vals):
        asymm_2_var += (((u[i] - v[i])**asymms_exp) - asymm_2_mean)**2

    asymm_2_var /= n_vals

    return asymm_2_var


cpdef DT_D get_asymm_2_skew(
        DT_D[:] u, DT_D[:] v, DT_D asymm_2_mean, DT_D asymm_2_var) except +:

    cdef:
        Py_ssize_t i, n_vals
        DT_D asymm_2_skew

    n_vals = u.shape[0]

    asymm_2_skew = 0.0
    for i in range(n_vals):
        asymm_2_skew += (((u[i] - v[i])**asymms_exp) - asymm_2_mean)**3

    asymm_2_skew /= n_vals

    asymm_2_skew /= asymm_2_var**1.5

    return asymm_2_skew


cpdef np.ndarray get_distances_from_vector_nd(
        double[:, ::1] pts,
        double[::1] vec_beg,
        double[::1] vec_end,) except +:

    cdef:
        Py_ssize_t i, j, n_pts, n_dims

        double vec_dir_dot

        double[:] vec_dir, rel_dists

        double[:, ::1] pts_dists

    n_pts = pts.shape[0]
    n_dims = pts.shape[1]

    vec_dir = np.empty(n_dims, dtype=np.float64)
    rel_dists = np.empty(n_pts, dtype=np.float64)
    pts_dists = np.empty((n_pts, n_dims), dtype=np.float64, order='c')

    vec_dir_dot = 0.0
    for j in range(n_dims):
        vec_dir[j] = vec_end[j] - vec_beg[j]
        vec_dir_dot += vec_dir[j] ** 2

    for i in range(n_pts):

        rel_dists[i] = 0.0
        for j in range(n_dims):
            rel_dists[i] += (pts[i, j] - vec_beg[j]) * vec_dir[j]

        rel_dists[i] /= vec_dir_dot

    for i in range(n_pts):
        for j in range(n_dims):
            pts_dists[i, j] = pts[i, j] - (
                (rel_dists[i] * vec_dir[j]) + vec_beg[j])

    return np.asarray(pts_dists)


cpdef np.ndarray get_asymms_nd_v2_raw_cy(
        double[:, ::1] probs) except +:

    cdef:
        Py_ssize_t i, j, k, n_probs, n_dims, n_asymms

        double vec_dir_dot

        double[:] vec_beg, vec_end, vec_dir, rel_dists

        double[:, ::1] probs_dists, asymm_vecs

    n_probs = probs.shape[0]
    n_dims = probs.shape[1]
    n_asymms = n_dims + 1

    vec_beg = np.empty(n_dims, dtype=np.float64)
    vec_end = np.empty(n_dims, dtype=np.float64)
    vec_dir = np.empty(n_dims, dtype=np.float64)
    rel_dists = np.empty(n_probs, dtype=np.float64)
    probs_dists = np.empty((n_probs, n_dims), dtype=np.float64, order='c')
    asymm_vecs = np.empty((n_asymms, n_dims), dtype=np.float64, order='c')

    for k in range(n_asymms):

        if k < n_dims:
            for j in range(n_dims):
    
                if j == k:
                    vec_end[j] = 1.0
    
                else:
                    vec_end[j] = 0.0
    
                vec_beg[j] = 1.0 - vec_end[j]
        else:
            for j in range(n_dims):
                vec_beg[j] = 0.0
                vec_end[j] = 1.0

        vec_dir_dot = 0.0
        for j in range(n_dims):
            vec_dir[j] = vec_end[j] - vec_beg[j]
            vec_dir_dot += vec_dir[j] ** 2

        for i in range(n_probs):

            rel_dists[i] = 0.0
            for j in range(n_dims):
                rel_dists[i] += (probs[i, j] - vec_beg[j]) * vec_dir[j]

            rel_dists[i] /= vec_dir_dot

        for i in range(n_probs):
            for j in range(n_dims):
                probs_dists[i, j] = probs[i, j] - (
                    (rel_dists[i] * vec_dir[j]) + vec_beg[j])

        for j in range(n_dims):
            asymm_vecs[k, j] = 0.0

        for i in range(n_probs):
            for j in range(n_dims):
                asymm_vecs[k, j] += probs_dists[i, j] ** 3

        for j in range(n_dims):
            asymm_vecs[k, j] /= n_probs

    return np.asarray(asymm_vecs)