cmake_minimum_required(VERSION 4.2)
project(dsp_hw1)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp")

add_executable(
    dsp_hw1
    src/main.cpp
    src/parallel/k_way_mergesort.cpp
    src/parallel/k_way_mergesort.hpp
    src/sequential/k_way_mergesort.cpp
    src/sequential/k_way_mergesort.hpp
    src/util.hpp
)
