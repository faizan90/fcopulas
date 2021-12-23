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


cpdef void fill_bin_idxs_ts(
        const DT_D[::1] probs,
              DT_UL[::1] bins_ts, 
        const DT_UL n_bins) except +

cpdef void fill_bin_dens_1d(
        const DT_UL[::1] bins_ts,
              DT_D[::1] bins_dens) except +

cpdef void fill_bin_dens_2d(
        const DT_UL[::1] bins_ts_x,
        const DT_UL[::1] bins_ts_y,
              DT_D[:, ::1] bins_dens_xy) except +

cpdef void fill_etpy_lcl_ts(
        const DT_UL[::1] bins_ts_x,
        const DT_UL[::1] bins_ts_y,
        const DT_D[::1] bins_dens_x,
        const DT_D[::1] bins_dens_y,
              DT_D[::1] etpy_ts,
        const DT_D[:, ::1] bins_dens_xy) except +

cpdef void fill_bi_var_cop_dens(
        DT_D[:] x_probs, DT_D[:] y_probs, DT_D[:, ::1] emp_dens_arr) except +

cpdef void fill_cumm_dist_from_bivar_emp_dens(
        DT_D[:, ::1] emp_dens_arr, DT_D[:, ::1] cum_emp_dens_arr) except +

cpdef tuple sample_from_2d_ecop(
        const DT_D[:, ::1] ecop_dens_arr,
        const DT_UL nvs,
        const DT_UL init_idx) except +

cpdef void fill_bi_var_cop_dens(
        DT_D[:] x_probs, DT_D[:] y_probs, DT_D[:, ::1] emp_dens_arr) except +

cpdef void fill_cumm_dist_from_bivar_emp_dens(
        DT_D[:, ::1] emp_dens_arr, DT_D[:, ::1] cum_emp_dens_arr) except +

cpdef DT_D get_etpy_min(DT_UL n_bins) except +

cpdef DT_D get_etpy_max(DT_UL n_bins) except +

cpdef np.ndarray get_ecop_nd(
        const double[:, ::1] probs, 
        const unsigned long long n_bins) except +

cpdef np.ndarray get_ecop_nd_empirical(
        const unsigned long long[:, ::1] ranks) except +

cpdef np.ndarray get_ecop_nd_empirical_sorted(
        const unsigned long long[:, ::1] ranks_sorted) except +
