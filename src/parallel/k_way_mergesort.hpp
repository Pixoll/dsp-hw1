#pragma once

#include <vector>

void parallel_k_way_mergesort(std::vector<int> &array, int k, int threshold);
void parallel_k_way_mergesort(std::vector<int> &array, int k); // NOLINT(*-narrowing-conversions)
