cmake_minimum_required(VERSION 4.2)
project(dsp_hw1)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp")

find_package(OpenMP REQUIRED)

add_executable(
    dsp_hw1
    src/main.cpp
    src/parallel/k_way_mergesort.cpp
    src/parallel/k_way_mergesort.hpp
    src/parallel/mergesort.cpp
    src/parallel/mergesort.hpp
    src/parallel/ranks_mergesort.cpp
    src/parallel/ranks_mergesort.hpp
    src/parallel/ranks_k_way_mergesort.cpp
    src/parallel/ranks_k_way_mergesort.hpp
    src/parallel/merge/merge.hpp
    src/parallel/merge/merge.cpp
    src/parallel/merge/k_merge.hpp
    src/parallel/merge/k_merge.cpp
    src/sequential/k_way_mergesort.cpp
    src/sequential/k_way_mergesort.hpp
    src/sequential/mergesort.cpp
    src/sequential/mergesort.hpp
    src/util.hpp
)

target_link_libraries(dsp_hw1 PUBLIC OpenMP::OpenMP_CXX)
