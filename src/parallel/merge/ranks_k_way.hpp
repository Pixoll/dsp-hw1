#pragma once

#include <vector>

#include "../../util.hpp"

void parallel_ranks_k_way_merge(
    const std::vector<int> &array,
    std::vector<int> &helper,
    const std::vector<Range> &partitions,
    int start
);
