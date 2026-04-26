#pragma once

#include <vector>

void parallel_merge_ranks(const std::vector<int> &src, std::vector<int> &dest,
                          int l1, int r1, int l2, int r2, int out_pos);