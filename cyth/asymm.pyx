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
        DT_D amax

    a_max = (
        0.5 * (1 - scorr)) * (1 - ((0.5 * (1 - scorr)) ** (1.0 / 3.0)))

    return a_max


cpdef DT_D get_asymm_2_max(DT_D scorr) except +:

    cdef:
        DT_D amax

    a_max = (
        0.5 * (1 + scorr)) * (1 - ((0.5 * (1 + scorr)) ** (1.0 / 3.0)))

    return a_max