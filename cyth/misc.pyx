# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True


cdef int chk_uexp_ovf(
        const unsigned long long base,
        const unsigned long long exponent) nogil:

    cdef:
        unsigned long long i, pre_val, post_val, result

    pre_val = base
    result = 0
    for i in range(2, exponent + 1):
        post_val = base**i

        if post_val > pre_val:
            pre_val = post_val
            result = 0
        
        else:
            result = 1
            break

    return result
