#include "../inc/duda.h"

#include "../inc/drray.h"

#include "../inc/kernel.h"


#define tx ( threadIdx.x )
#define ty ( threadIdx.y )



st_DeviceArray DeviceArray( uint32_t row, uint32_t col )
{
    st_DeviceArray dary;

    dary.row = row;
    dary.col = col;

    uint32_t size = row * col;

    cudaMalloc( &dary.ptr, sizeof( float ) * size );

    cudaMemset(  dary.ptr, 0.0, sizeof( float ) * size );

    return dary;
}


st_DeviceArray DeviceIdentity( uint32_t ndim )
{
    st_DeviceArray dary;

    dary.row = ndim;
    dary.col = ndim;

    uint32_t size = ndim * ndim;

    cudaMalloc( &dary.ptr, sizeof( float ) * size );

    _Identity_kernel <<<1,ndim>>> ( dary.ptr, dary.row, dary.col );

    cudaDeviceSynchronize();

    return dary;
}


error_DUDA Clean( st_DeviceArray* dary )
{
    cudaFree( dary->ptr );

    return DUDA_SUCCESS;
}


error_DUDA Print( st_DeviceArray* dary )
{
    uint32_t size = dary->row * dary->col;

    st_Array1D ary1D = Array1D( size );

    cudaMemcpy( ary1D.ptr, dary->ptr, sizeof( float ) * size, cudaMemcpyDeviceToHost );

    st_Array ary = Array2Array1D( &ary1D, dary->row, dary->col );

    if ( Print( &ary ) != ARR_SUCCESS ) return DUDA_FAILURE;

    Clean( &ary );
    Clean( &ary1D );

    return DUDA_SUCCESS;
}


error_DUDA ElementwiseAdd( st_DeviceArray* OUTPUT, const st_DeviceArray* INPUT1, const st_DeviceArray* INPUT2 )
{
    if ( ( OUTPUT->row != INPUT1->row ) || ( OUTPUT->row != INPUT2->row ) ) return DUDA_UNMATCH;

    if ( ( OUTPUT->col != INPUT1->col ) || ( OUTPUT->col != INPUT2->col ) ) return DUDA_UNMATCH;

    dim3 blockDim( OUTPUT->row, OUTPUT->col );

    _ElementwiseAdd_kernel <<<1, blockDim>>> ( OUTPUT->ptr, INPUT1->ptr, INPUT2->ptr, OUTPUT->row, OUTPUT->col );

    cudaDeviceSynchronize();

    return DUDA_SUCCESS;
}


error_DUDA ElementwiseSub( st_DeviceArray* OUTPUT, const st_DeviceArray* INPUT1, const st_DeviceArray* INPUT2 )
{
    if ( ( OUTPUT->row != INPUT1->row ) || ( OUTPUT->row != INPUT2->row ) ) return DUDA_UNMATCH;

    if ( ( OUTPUT->col != INPUT1->col ) || ( OUTPUT->col != INPUT2->col ) ) return DUDA_UNMATCH;

    dim3 blockDim( OUTPUT->row, OUTPUT->col );

    _ElementwiseSub_kernel <<<1, blockDim>>> ( OUTPUT->ptr, INPUT1->ptr, INPUT2->ptr, OUTPUT->row, OUTPUT->col );

    cudaDeviceSynchronize();

    return DUDA_SUCCESS;
}


error_DUDA ElementwiseMul( st_DeviceArray* OUTPUT, const st_DeviceArray* INPUT1, const st_DeviceArray* INPUT2 )
{
    if ( ( OUTPUT->row != INPUT1->row ) || ( OUTPUT->row != INPUT2->row ) ) return DUDA_UNMATCH;

    if ( ( OUTPUT->col != INPUT1->col ) || ( OUTPUT->col != INPUT2->col ) ) return DUDA_UNMATCH;

    dim3 blockDim( OUTPUT->row, OUTPUT->col );

    _ElementwiseMul_kernel <<<1, blockDim>>> ( OUTPUT->ptr, INPUT1->ptr, INPUT2->ptr, OUTPUT->row, OUTPUT->col );

    cudaDeviceSynchronize();

    return DUDA_SUCCESS;
}


error_DUDA ElementwiseSub( st_DeviceArray* OUTPUT, const st_DeviceArray* INPUT1, const st_DeviceArray* INPUT2 )
{
    if ( ( OUTPUT->row != INPUT1->row ) || ( OUTPUT->row != INPUT2->row ) ) return DUDA_UNMATCH;

    if ( ( OUTPUT->col != INPUT1->col ) || ( OUTPUT->col != INPUT2->col ) ) return DUDA_UNMATCH;

    dim3 blockDim( OUTPUT->row, OUTPUT->col );

    _ElementwiseSub_kernel <<<1, blockDim>>> ( OUTPUT->ptr, INPUT1->ptr, INPUT2->ptr, OUTPUT->row, OUTPUT->col );

    cudaDeviceSynchronize();

    return DUDA_SUCCESS;
}
