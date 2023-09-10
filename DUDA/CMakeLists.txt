cmake_minimum_required( VERSION 3.10 )

project( DUDA )

add_library( DUDA src/drray.cpp src/duda.cu src/kernel.cu )

target_include_directories( DUDA PUBLIC inc )