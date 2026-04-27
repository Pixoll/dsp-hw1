#pragma once

#include <vector>

void parallel_ranks_merge(
    const std::vector<int> &array,
    std::vector<int> &helper,
    int l1,
    int r1,
    int l2,
    int r2,
    int start
);
