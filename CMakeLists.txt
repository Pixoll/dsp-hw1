cmake_minimum_required(VERSION 3.28)
project(dsp_hw1)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenMP REQUIRED)

add_executable(dsp_hw1 main.cpp secuencial/mergesort.cpp paralelo/mergesort_paralelo.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(dsp_hw1 PUBLIC OpenMP::OpenMP_CXX)
endif()