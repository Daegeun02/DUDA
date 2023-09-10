cmake_minimum_required( VERSION 3.10 )

project( run LANGUAGES CXX CUDA )

SET( CMAKE_CUDA_COMPILER /usr/local/cuda-11.4/bin/nvcc )

add_subdirectory( DUDA )

add_executable( run main.cu )

target_link_libraries( run PUBLIC DUDA )