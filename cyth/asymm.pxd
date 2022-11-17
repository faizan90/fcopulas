# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

ctypedef double DT_D
ctypedef unsigned long DT_UL

import numpy as np
cimport numpy as np

cpdef float asymms_exp

cpdef float get_asymms_exp()

cpdef tuple get_asymms_sample(DT_D[:] u, DT_D[:] v) except +

cpdef DT_D get_asymm_1_sample(DT_D[:] u, DT_D[:] v) except +

cpdef DT_D get_asymm_2_sample(DT_D[:] u, DT_D[:] v) except +

cpdef DT_D get_asymm_1_max(DT_D scorr) except +

cpdef DT_D get_asymm_2_max(DT_D scorr) except +

cpdef DT_D get_asymm_1_var(DT_D[:] u, DT_D[:] v, DT_D asymm_1_mean) except +

cpdef DT_D get_asymm_1_skew(
        DT_D[:] u, DT_D[:] v, DT_D asymm_1_mean, DT_D asymm_1_var) except +

cpdef DT_D get_asymm_2_var(DT_D[:] u, DT_D[:] v, DT_D asymm_2_mean) except +

cpdef DT_D get_asymm_2_skew(
        DT_D[:] u, DT_D[:] v, DT_D asymm_1_mean, DT_D asymm_2_var) except +

cpdef np.ndarray get_distances_from_vector_nd(
        double[:, ::1] pts,
        double[::1] vec_beg,
        double[::1] vec_end,) except +

cpdef void fill_probs_dists_v2_cy(
        const double[:, ::1] probs, 
        double[:, ::1] probs_dists, 
        Py_ssize_t asymm_idx) except +

cpdef np.ndarray get_asymms_nd_v2_raw_cy(
        const double[:, ::1] probs) except +
