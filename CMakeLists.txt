cmake_minimum_required(VERSION 4.2)
project(dsp_hw1)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp")

find_package(OpenMP REQUIRED)

add_executable(
    dsp_hw1
    src/main.cpp
        src/parallel_merge_sort/k_way_mergesort.cpp
        src/parallel_merge_sort/k_way_mergesort.hpp
        src/parallel_merge_sort/mergesort.cpp
        src/parallel_merge_sort/mergesort.hpp
        src/paralllel_merge/parallel_merge.hpp
        src/paralllel_merge/parallel_merge.cpp
        src/sequential_merge_sort/k_way_mergesort.cpp
        src/sequential_merge_sort/k_way_mergesort.hpp
        src/sequential_merge_sort/mergesort.cpp
        src/sequential_merge_sort/mergesort.hpp
    src/util.hpp
        src/parallel_merge_sort/parallel_mergesort_ranks.cpp
        src/parallel_merge_sort/parallel_mergesort_ranks.hpp
        src/paralllel_merge/parallel_k_merge.cpp
        src/paralllel_merge/parallel_k_merge.hpp
        src/parallel_merge_sort/k_way_mergesort_ranks.cpp
        src/parallel_merge_sort/k_way_mergesort_ranks.hpp
)

target_link_libraries(dsp_hw1 PUBLIC OpenMP::OpenMP_CXX)
