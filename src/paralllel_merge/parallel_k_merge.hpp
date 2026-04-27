#pragma once

#include <vector>

struct Range {
    int start;
    int end;
    int size() const;
};

void parallel_k_merge_ranks(const std::vector<int> &src, std::vector<int> &dest, 
                            std::vector<Range> partitions, int out_pos);