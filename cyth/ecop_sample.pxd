# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

ctypedef double DT_D
ctypedef unsigned long long DT_UL

cpdef tuple get_cond_idx2_2d_ecop(
        const DT_D[:, ::1] ecop_dens_arr,
        const DT_UL[::1] idxs_freqs,
        const DT_D u1) except +
