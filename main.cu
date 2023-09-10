#include <iostream>

#include "./DUDA/inc/duda.h"
#include "./DUDA/inc/kernel.h"


st_DeviceArray I1;
st_DeviceArray I2;
st_DeviceArray I3;



int main( void )
{
    I1 = DeviceIdentity( 5 );
    I2 = DeviceIdentity( 5 );
    I3 = DeviceArray( 5, 5 );

    ElementwiseAdd( &I3, &I1, &I2 );

    Print( &I1 );
    Print( &I2 );
    Print( &I3 );

    Clean( &I1 );
    Clean( &I2 );
    Clean( &I3 );

    return 0;
}