# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

ctypedef double DT_D
ctypedef unsigned long DT_UL


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
