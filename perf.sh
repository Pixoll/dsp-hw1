#!/bin/bash -x

cd build
sudo perf stat -e cycles,instructions,cache-references,cache-misses ./dsp_hw1 sequential_mergesort
sudo perf stat -e cycles,instructions,cache-references,cache-misses ./dsp_hw1 sequential_k_way_mergesort
sudo perf stat -e cycles,instructions,cache-references,cache-misses ./dsp_hw1 parallel_mergesort
sudo perf stat -e cycles,instructions,cache-references,cache-misses ./dsp_hw1 parallel_k_way_mergesort
sudo perf stat -e cycles,instructions,cache-references,cache-misses ./dsp_hw1 parallel_ranks_mergesort
sudo perf stat -e cycles,instructions,cache-references,cache-misses ./dsp_hw1 parallel_ranks_k_way_mergesort
