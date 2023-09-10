#pragma once

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdint.h>


typedef struct {
    double **ptr;
    uint32_t row;
    uint32_t col;
} st_Array;


typedef struct {
    float    *ptr;
    uint32_t size;
} st_Array1D;


typedef enum {
    ARR_FAILURE = 0,
    ARR_SUCCESS = 1,
    ARR_ZERODIV = 2,
    ARR_UNMATCH = 3
} error_ARR;


st_Array Array( uint32_t, uint32_t );
st_Array Identity( uint32_t );

st_Array1D Array1D( uint32_t );
st_Array Array2Array1D( const st_Array1D*, const uint32_t, const uint32_t );

error_ARR Memset( st_Array*, double );
error_ARR Clean( st_Array* );
error_ARR Clean( st_Array1D* );

error_ARR Print( st_Array* );

error_ARR ElementwiseAdd( st_Array*, const st_Array*, const st_Array* );
error_ARR ElementwiseSub( st_Array*, const st_Array*, const st_Array* );
error_ARR ElementwiseMul( st_Array*, const st_Array*, const st_Array* );
error_ARR ElementwiseDiv( st_Array*, const st_Array*, const st_Array* );

error_ARR ElementwiseAdd( st_Array*, const st_Array* );
error_ARR ElementwiseSub( st_Array*, const st_Array* );
error_ARR ElementwiseMul( st_Array*, const st_Array* );
error_ARR ElementwiseDiv( st_Array*, const st_Array* );

error_ARR ElementwiseAdd( st_Array*, double );
error_ARR ElementwiseSub( st_Array*, double );
error_ARR ElementwiseMul( st_Array*, double );
error_ARR ElementwiseDiv( st_Array*, double );

error_ARR MatrixMultiply( st_Array*, const st_Array*, const st_Array* );

error_ARR MatrixTranspose( st_Array*, const st_Array* );

error_ARR MatrixInverse( st_Array*, const st_Array* );

error_ARR MatrixSolve( st_Array*, st_Array*, const st_Array*, const st_Array* );

error_ARR SymmetricMatrixSolve( st_Array*, const st_Array*, const st_Array* );

error_ARR CholeskiDecomposition( st_Array*, const st_Array* );

error_ARR BackSubsitution( st_Array*, const st_Array*, const st_Array* );
error_ARR ForwSubsitution( st_Array*, const st_Array*, const st_Array* );