#pragma once

#include <vector>

void parallel_mergesort(std::vector<int> &array, int threshold);
void parallel_mergesort(std::vector<int> &array); // NOLINT(*-narrowing-conversions)