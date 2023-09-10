#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"



__global__ void _Identity_kernel( float*, uint32_t, uint32_t );

__global__ void _ElementwiseAdd_kernel( float*, float*, float*, uint32_t, uint32_t );
__global__ void _ElementwiseSub_kernel( float*, float*, float*, uint32_t, uint32_t );
__global__ void _ElementwiseMul_kernel( float*, float*, float*, uint32_t, uint32_t );
__global__ void _ElementwiseDiv_kernel( float*, float*, float*, uint32_t, uint32_t );