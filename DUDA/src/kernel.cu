#include "../inc/kernel.h"


#define tx ( threadIdx.x )
#define ty ( threadIdx.y )


__global__ void _Identity_kernel( float* ptr, uint32_t row, uint32_t col )
{
    uint32_t idx = tx * col + tx;

    ptr[idx] = 1.0;

    __syncthreads();
}


__global__ void _ElementwiseAdd_kernel( float* OUTPUT, float* INPUT1, float* INPUT2, uint32_t row, uint32_t col )
{
    uint32_t idx = tx * col + ty;

    OUTPUT[idx] = INPUT1[idx] + INPUT2[idx];

    __syncthreads();
}


__global__ void _ElementwiseSub_kernel( float* OUTPUT, float* INPUT1, float* INPUT2, uint32_t row, uint32_t col )
{
    uint32_t idx = tx * col + ty;

    OUTPUT[idx] = INPUT1[idx] - INPUT2[idx];

    __syncthreads();
}


__global__ void _ElementwiseMul_kernel( float* OUTPUT, float* INPUT1, float* INPUT2, uint32_t row, uint32_t col )
{
    uint32_t idx = tx * col + ty;

    OUTPUT[idx] = INPUT1[idx] * INPUT2[idx];

    __syncthreads();
}


__global__ void _ElementwiseDiv_kernel( float* OUTPUT, float* INPUT1, float* INPUT2, uint32_t row, uint32_t col )
{
    uint32_t idx = tx * col + ty;

    if ( INPUT2[idx] != 0.0 )
    {
        OUTPUT[idx] = INPUT1[idx] / INPUT2[idx];
    }
    else
    {
        OUTPUT[idx] = 1e+38;
    }

    __syncthreads();
}