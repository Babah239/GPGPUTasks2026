#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело оставляйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! ПЛЫВИТЕ ПО ТЕЧЕНИЮ!! ПОРЯДОК!!! ХАЙП!11!!1
    __global const uint* pow2_sum,
    __global const uint* input, 
    __global       uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    unsigned int pow2)
{
    uint x = get_global_id(0), answer=0;
    if (x < n) {
        for (uint i = 1; i < pow2; ++i)
            if ((x+1) & (1 << i))
                answer += pow2_sum[(1 << pow2 - i) + (x+1 >> i)-1];
        if ((x+1) & 1)
            answer += input[x];
        prefix_sum_accum[x] = answer;
    }
}
