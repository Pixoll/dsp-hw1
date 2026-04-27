#pragma once

#include <vector>

#include "../../util.hpp"

void parallel_k_merge_ranks(
    const std::vector<int> &array,
    std::vector<int> &helper,
    const std::vector<Range> &partitions,
    int start
);
