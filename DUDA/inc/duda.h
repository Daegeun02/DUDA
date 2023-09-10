#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"



typedef struct {
    float   *ptr;
    uint32_t row;
    uint32_t col;
} st_DeviceArray;


typedef enum {
    DUDA_FAILURE = 0,
    DUDA_SUCCESS = 1,
    DUDA_ZERODIV = 2,
    DUDA_UNMATCH = 3
} error_DUDA;


st_DeviceArray DeviceArray( uint32_t, uint32_t );
st_DeviceArray DeviceIdentity( uint32_t );

error_DUDA Clean( st_DeviceArray* );

error_DUDA Print( st_DeviceArray* );

error_DUDA ElementwiseAdd( st_DeviceArray*, const st_DeviceArray*, const st_DeviceArray* );
error_DUDA ElementwiseSub( st_DeviceArray*, const st_DeviceArray*, const st_DeviceArray* );
error_DUDA ElementwiseMul( st_DeviceArray*, const st_DeviceArray*, const st_DeviceArray* );
error_DUDA ElementwiseDiv( st_DeviceArray*, const st_DeviceArray*, const st_DeviceArray* );