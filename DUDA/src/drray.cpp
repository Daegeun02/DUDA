#include "../inc/drray.h"



st_Array Array( uint32_t row, uint32_t col )
{
    st_Array ary;

    ary.row = row;
    ary.col = col;

    ary.ptr = new double*[row];

    for ( uint32_t idx = 0; idx < row; idx++ )
    {
        ary.ptr[idx] = new double[col];
    }

    Memset( &ary, 0.0 );

    return ary;
}

st_Array Identity( uint32_t ndim )
{
    st_Array ary = Array( ndim, ndim );

    for ( uint32_t i = 0; i < ndim; i++ )
    {
        ary.ptr[i][i] = 1.0;
    }

    return ary;
}


st_Array1D Array1D( uint32_t size )
{
    st_Array1D ary;

    ary.size = size;

    ary.ptr = new float[size];

    for ( uint32_t idx = 0; idx < size; idx++ )
    {
        ary.ptr[idx] = 0.0;
    }

    return ary;
}


st_Array Array2Array1D( const st_Array1D* ary1D, const uint32_t row, const uint32_t col )
{
    st_Array ary = Array( row, col );

    uint32_t idx;

    double** ptr   = ary.ptr;
    float*   ptr1D = ary1D->ptr;

    for ( uint32_t i = 0; i < row; i++ )
    {
        for ( uint32_t j = 0; j < col; j++ )
        {
            idx = i * col + j;

            ptr[i][j] = ptr1D[idx];
        }
    }

    return ary;
}


error_ARR Memset( st_Array* INPUT, double value )
{
    double **ptr = INPUT->ptr;
    uint32_t row = INPUT->row;
    uint32_t col = INPUT->col;

    for ( uint32_t i = 0; i < row; i++ )
    {
        for ( uint32_t j = 0; j < col; j++ )
        {
            ptr[i][j] = value;
        }
    }

    return ARR_SUCCESS;
}


error_ARR Clean( st_Array* INPUT )
{
    for ( uint32_t idx = 0; idx < INPUT->row; idx++ )
    {
        delete[] INPUT->ptr[idx];
    }

    delete[] INPUT->ptr;

    return ARR_SUCCESS;
}


error_ARR Clean( st_Array1D* INPUT )
{
    delete[] INPUT->ptr;

    return ARR_SUCCESS;
}


error_ARR Print( st_Array* INPUT )
{
    double **ptr = INPUT->ptr;
    uint32_t row = INPUT->row;
    uint32_t col = INPUT->col;

    for ( uint32_t i = 0; i < row; i++ )
    {
        for ( uint32_t j = 0; j < col; j++ )
        {
            printf( "%f ", ptr[i][j] );
        }

        printf( "\n" );
    }

    printf( "\n" );

    return ARR_SUCCESS;
}


error_ARR ElementwiseAdd( st_Array* OUTPUT, const st_Array* INPUT1, const st_Array* INPUT2 )
{
    // size check
    if ( ( OUTPUT->row != INPUT1->row ) || ( OUTPUT->row != INPUT2->row ) ) return ARR_FAILURE;
    if ( ( OUTPUT->col != INPUT1->col ) || ( OUTPUT->col != INPUT2->col ) ) return ARR_FAILURE;

    uint32_t row = OUTPUT->row;
    uint32_t col = OUTPUT->col;

    double **ptrO = OUTPUT->ptr;
    double **ptr1 = INPUT1->ptr;
    double **ptr2 = INPUT2->ptr;

    for ( uint32_t i = 0; i < row; i++ )
    {
        for ( uint32_t j = 0; j < col; j++ )
        {
            ptrO[i][j] = ptr1[i][j] + ptr2[i][j];
        }
    }

    return ARR_SUCCESS;
}

error_ARR ElementwiseSub( st_Array* OUTPUT, const st_Array* INPUT1, const st_Array* INPUT2 )
{
    // size check
    if ( ( OUTPUT->row != INPUT1->row ) || ( OUTPUT->row != INPUT2->row ) ) return ARR_FAILURE;
    if ( ( OUTPUT->col != INPUT1->col ) || ( OUTPUT->col != INPUT2->col ) ) return ARR_FAILURE;

    uint32_t row = OUTPUT->row;
    uint32_t col = OUTPUT->col;

    double **ptrO = OUTPUT->ptr;
    double **ptr1 = INPUT1->ptr;
    double **ptr2 = INPUT2->ptr;

    for ( uint32_t i = 0; i < row; i++ )
    {
        for ( uint32_t j = 0; j < col; j++ )
        {
            ptrO[i][j] = ptr1[i][j] - ptr2[i][j];
        }
    }

    return ARR_SUCCESS;
}

error_ARR ElementwiseMul( st_Array* OUTPUT, const st_Array* INPUT1, const st_Array* INPUT2 )
{
    // size check
    if ( ( OUTPUT->row != INPUT1->row ) || ( OUTPUT->row != INPUT2->row ) ) return ARR_FAILURE;
    if ( ( OUTPUT->col != INPUT1->col ) || ( OUTPUT->col != INPUT2->col ) ) return ARR_FAILURE;

    uint32_t row = OUTPUT->row;
    uint32_t col = OUTPUT->col;

    double **ptrO = OUTPUT->ptr;
    double **ptr1 = INPUT1->ptr;
    double **ptr2 = INPUT2->ptr;

    for ( uint32_t i = 0; i < row; i++ )
    {
        for ( uint32_t j = 0; j < col; j++ )
        {
            ptrO[i][j] = ptr1[i][j] * ptr2[i][j];
        }
    }

    return ARR_SUCCESS;
}


error_ARR ElementwiseDiv( st_Array* OUTPUT, const st_Array* INPUT1, const st_Array* INPUT2 )
{
    // size check
    if ( ( OUTPUT->row != INPUT1->row ) || ( OUTPUT->row != INPUT2->row ) ) return ARR_FAILURE;
    if ( ( OUTPUT->col != INPUT1->col ) || ( OUTPUT->col != INPUT2->col ) ) return ARR_FAILURE;

    uint32_t row = OUTPUT->row;
    uint32_t col = OUTPUT->col;

    double **ptrO = OUTPUT->ptr;
    double **ptr1 = INPUT1->ptr;
    double **ptr2 = INPUT2->ptr;

    for ( uint32_t i = 0; i < row; i++ )
    {
        for ( uint32_t j = 0; j < col; j++ )
        {
            if ( ptr2[i][j] == 0.0 ) return ARR_ZERODIV;

            ptrO[i][j] = ptr1[i][j] / ptr2[i][j];
        }
    }

    return ARR_SUCCESS;
}


error_ARR ElementwiseAdd( st_Array* OUTPUT, const st_Array* INPUT0 )
{
    // size check
    if ( ( OUTPUT->row != INPUT0->row ) || ( OUTPUT->col != INPUT0->col ) ) return ARR_FAILURE;

    uint32_t row = OUTPUT->row;
    uint32_t col = OUTPUT->col;

    double **ptrO = OUTPUT->ptr;
    double **ptr0 = INPUT0->ptr;

    for ( uint32_t i = 0; i < row; i++ )
    {
        for ( uint32_t j = 0; j < col; j++ )
        {
            ptrO[i][j] += ptr0[i][j];
        }
    }

    return ARR_SUCCESS;
}


error_ARR ElementwiseSub( st_Array* OUTPUT, const st_Array* INPUT0 )
{
    // size check
    if ( ( OUTPUT->row != INPUT0->row ) || ( OUTPUT->col != INPUT0->col ) ) return ARR_FAILURE;

    uint32_t row = OUTPUT->row;
    uint32_t col = OUTPUT->col;

    double **ptrO = OUTPUT->ptr;
    double **ptr0 = INPUT0->ptr;

    for ( uint32_t i = 0; i < row; i++ )
    {
        for ( uint32_t j = 0; j < col; j++ )
        {
            ptrO[i][j] -= ptr0[i][j];
        }
    }

    return ARR_SUCCESS;
}


error_ARR ElementwiseMul( st_Array* OUTPUT, const st_Array* INPUT0 )
{
    // size check
    if ( ( OUTPUT->row != INPUT0->row ) || ( OUTPUT->col != INPUT0->col ) ) return ARR_FAILURE;

    uint32_t row = OUTPUT->row;
    uint32_t col = OUTPUT->col;

    double **ptrO = OUTPUT->ptr;
    double **ptr0 = INPUT0->ptr;

    for ( uint32_t i = 0; i < row; i++ )
    {
        for ( uint32_t j = 0; j < col; j++ )
        {
            ptrO[i][j] *= ptr0[i][j];
        }
    }

    return ARR_SUCCESS;
}


error_ARR ElementwiseDiv( st_Array* OUTPUT, const st_Array* INPUT0 )
{
    // size check
    if ( ( OUTPUT->row != INPUT0->row ) || ( OUTPUT->col != INPUT0->col ) ) return ARR_FAILURE;

    uint32_t row = OUTPUT->row;
    uint32_t col = OUTPUT->col;

    double **ptrO = OUTPUT->ptr;
    double **ptr0 = INPUT0->ptr;

    for ( uint32_t i = 0; i < row; i++ )
    {
        for ( uint32_t j = 0; j < col; j++ )
        {
            if ( ptr0[i][j] == 0.0 ) return ARR_ZERODIV;

            ptrO[i][j] /= ptr0[i][j];
        }
    }

    return ARR_SUCCESS;
}


error_ARR ElementwiseAdd( st_Array* OUTPUT, double value )
{
    uint32_t row = OUTPUT->row;
    uint32_t col = OUTPUT->col;

    double **ptrO = OUTPUT->ptr;

    for ( uint32_t i = 0; i < row; i++ )
    {
        for ( uint32_t j = 0; j < col; j++ )
        {
            ptrO[i][j] += value;
        }
    }

    return ARR_SUCCESS;
}


error_ARR ElementwiseSub( st_Array* OUTPUT, double value )
{
    uint32_t row = OUTPUT->row;
    uint32_t col = OUTPUT->col;

    double **ptrO = OUTPUT->ptr;

    for ( uint32_t i = 0; i < row; i++ )
    {
        for ( uint32_t j = 0; j < col; j++ )
        {
            ptrO[i][j] -= value;
        }
    }

    return ARR_SUCCESS;
}


error_ARR ElementwiseMul( st_Array* OUTPUT, double value )
{
    uint32_t row = OUTPUT->row;
    uint32_t col = OUTPUT->col;

    double **ptrO = OUTPUT->ptr;

    for ( uint32_t i = 0; i < row; i++ )
    {
        for ( uint32_t j = 0; j < col; j++ )
        {
            ptrO[i][j] *= value;
        }
    }

    return ARR_SUCCESS;
}


error_ARR ElementwiseDiv( st_Array* OUTPUT, double value )
{
    if ( value == 0.0 ) return ARR_FAILURE;

    uint32_t row = OUTPUT->row;
    uint32_t col = OUTPUT->col;

    double **ptrO = OUTPUT->ptr;

    for ( uint32_t i = 0; i < row; i++ )
    {
        for ( uint32_t j = 0; j < col; j++ )
        {
            ptrO[i][j] /= value;
        }
    }

    return ARR_SUCCESS;
}


error_ARR MatrixMultiply( st_Array* OUTPUT, const st_Array* INPUT1, const st_Array* INPUT2 )
{
    // Multipliable check
    if ( INPUT1->col != INPUT2->row ) return ARR_FAILURE;
    // OUTPUT size check
    if ( ( OUTPUT->row != INPUT1->row ) || ( OUTPUT->col != INPUT2->col ) ) return ARR_FAILURE;

    double **ptrO = OUTPUT->ptr;
    double **ptr1 = INPUT1->ptr;
    double **ptr2 = INPUT2->ptr;

    for ( uint32_t i = 0; i < OUTPUT->row; i++ )
    {
        for ( uint32_t j = 0; j < OUTPUT->col; j++ )
        {
            ptrO[i][j] = 0.0;

            for ( uint32_t k = 0; k < INPUT1->col; k++ )
            {
                ptrO[i][j] += ptr1[i][k] * ptr2[k][j];
            }
        }
    }

    return ARR_SUCCESS;
}


error_ARR MatrixTranspose( st_Array* OUTPUT, const st_Array* INPUT0 )
{
    if ( ( OUTPUT->row != INPUT0->col ) || ( OUTPUT->col != INPUT0->row ) ) return ARR_FAILURE;

    double **ptrO = OUTPUT->ptr;
    double **ptrI = INPUT0->ptr;

    for ( uint32_t i = 0; i < OUTPUT->row; i++ )
    {
        for ( uint32_t j = 0; j < OUTPUT->col; j++ )
        {
            ptrO[i][j] = ptrI[j][i];
        }
    }

    return ARR_SUCCESS;
}


error_ARR MatrixInverse( st_Array* OUTPUT, const st_Array* INPUT0 )
{
    return ARR_FAILURE;
}


// Calculate right inverse of matrix INPUT1
// which is same that solve linear equation "INPUT1 @ OUTPUT = INPUT2"
error_ARR MatrixSolve( st_Array* OUTPUT, const st_Array* INPUT1, const st_Array* INPUT2 )
{
    // Multipliable check
    if ( INPUT1->col != OUTPUT->row ) return ARR_FAILURE;
    // OUTPUT size check
    if ( ( INPUT1->row != INPUT2->row ) || ( OUTPUT->col != INPUT2->col ) ) return ARR_FAILURE;

    double **ptrO = OUTPUT->ptr;
    double **ptr1 = INPUT1->ptr;
    double **ptr2 = INPUT2->ptr;

    return ARR_FAILURE;
}


// Calculate right inverse of matrix INPUT1
// which is same that solve linear equation "INPUT1 @ OUTPUT = INPUT2"
// this time INPUT1 is symmetric matrix
error_ARR SymmetricMatrixSolve( st_Array* OUTPUT, const st_Array* INPUT1, const st_Array* INPUT2 )
{
    st_Array buff1 = Array( INPUT1->row, INPUT1->col ); 
    st_Array buff2 = Array( INPUT1->row, INPUT1->col );
    st_Array buff3 = Array( INPUT2->row, INPUT2->col );

    if ( CholeskiDecomposition( &buff1, INPUT1 ) != ARR_SUCCESS )
    {
        printf( "Singularity Occured\n" );

        Clean( &buff1 );
        Clean( &buff2 );
        Clean( &buff3 );

        return ARR_FAILURE;
    }

    Print( &buff1 );

    MatrixTranspose( &buff2, &buff1 );

    if ( ForwSubsitution( &buff3, &buff1, INPUT2 ) != ARR_SUCCESS )
    {
        printf( "Size mismatch\n" );

        Clean( &buff1 );
        Clean( &buff2 );
        Clean( &buff3 );

        return ARR_FAILURE;
    }

    if ( BackSubsitution( OUTPUT, &buff2, &buff3 ) != ARR_SUCCESS )
    {
        printf( "Size mismatch\n" );

        Clean( &buff1 );
        Clean( &buff2 );
        Clean( &buff3 );

        return ARR_FAILURE;
    }

    Clean( &buff1 );
    Clean( &buff2 );
    Clean( &buff3 );

    return ARR_SUCCESS;
}


error_ARR CholeskiDecomposition( st_Array* OUTPUT, const st_Array* INPUT0 )
{
    // symmetric size check
    if ( INPUT0->row != INPUT0->col ) return ARR_FAILURE;
    // OUTPUT size check
    if ( ( OUTPUT->row != INPUT0->row ) || ( OUTPUT->col != INPUT0->col ) ) return ARR_FAILURE;

    uint32_t Ncol = OUTPUT->col;

    double sum = 0.0;

    double **ptrO = OUTPUT->ptr;
    double **ptr0 = INPUT0->ptr;

    Memset( OUTPUT, 0.0 );

    //Choleski Decompostion
    for ( uint32_t i = 0; i < INPUT0->row; i++ )
    {
        for ( uint32_t j = 0; j <= i; j++ )
        {
            sum = 0.0;

            if ( i == j )
            {
                for ( uint32_t k = 0; k < j; k++ )
                {
                    sum += ptrO[j][k] * ptrO[j][k];
                }

                ptrO[j][j] = sqrt( ptr0[j][j] - sum );

                // Singularity occured
                if ( ptrO[i][i] == 0.0 ) return ARR_FAILURE;
            }
            else
            {
                for ( uint32_t k = 0; k < j; k++ )
                {
                    sum += ptrO[i][k] * ptrO[j][k];
                }

                ptrO[i][j] = ( ptr0[i][j] - sum ) / ptrO[j][j];
            }
        }
    }

    return ARR_SUCCESS;
}


// "INPUT1 @ OUTPUT = INPUT2"
// INPUT1 is Upper Triangular matrix
error_ARR BackSubsitution( st_Array* OUTPUT, const st_Array* INPUT1, const st_Array* INPUT2 )
{
    if ( INPUT1->col != OUTPUT->row ) return ARR_FAILURE;

    if ( ( INPUT1->row != INPUT2->row ) || ( OUTPUT->col != INPUT2->col ) ) return ARR_FAILURE;

    double sum = 0.0;

    double **ptrO = OUTPUT->ptr;
    double **ptr1 = INPUT1->ptr;
    double **ptr2 = INPUT2->ptr;

    for ( uint32_t j = 0; j < OUTPUT->col; j++ )
    {
        for ( uint32_t i = OUTPUT->row - 1; i != 0xFFFFFFFF; i-- )
        {
            sum = 0.0;

            for ( uint32_t k = i+1; k < OUTPUT->row; k++ )
            {
                sum += ptr1[i][k] * ptrO[k][j];
            }

            ptrO[i][j] = ( ptr2[i][j] - sum ) / ptr1[i][i];
        }
    }

    return ARR_SUCCESS;
}


// "INPUT1 @ OUTPUT = INPUT2"
// INPUT1 is Lower Triangular matrix
error_ARR ForwSubsitution( st_Array* OUTPUT, const st_Array* INPUT1, const st_Array* INPUT2 )
{
    if ( INPUT1->col != OUTPUT->row ) return ARR_FAILURE;

    if ( ( INPUT1->row != INPUT2->row ) || ( OUTPUT->col != INPUT2->col ) ) return ARR_FAILURE;

    double sum = 0.0;

    double **ptrO = OUTPUT->ptr;
    double **ptr1 = INPUT1->ptr;
    double **ptr2 = INPUT2->ptr;

    for ( uint32_t j = 0; j < OUTPUT->col; j++ )
    {
        for ( uint32_t i = 0; i < OUTPUT->row; i++ )
        {
            sum = 0.0;

            for ( uint32_t k = 0; k < i; k++ )
            {
                sum += ptr1[i][k] * ptrO[k][j];
            }

            ptrO[i][j] = ( ptr2[i][j] - sum ) / ptr1[i][i];
        }
    }

    return ARR_SUCCESS;
}