#include <vector>
#include <algorithm>
#include <omp.h>
#include "../paralllel_merge/parallel_k_merge.hpp"

static constexpr int SORT_THRESHOLD = 4096;
static constexpr int MAX_DEPTH = 4;

static void divide(std::vector<int>& array, std::vector<int>& helper,
                   const int k, const int start, const int end, const int depth = 0) {
    const int size = end - start + 1;
    if (size <= 1) return;

    if (size < SORT_THRESHOLD) {
        std::sort(array.begin() + start, array.begin() + end + 1);
        return;
    }

    std::vector<Range> parts;
    int current = start;
    for (int i = 0; i < k && current <= end; i++) {
        int p_size = (end - current + 1) / (k - i);
        int p_end  = current + p_size - 1;
        parts.push_back({current, p_end});

        if (depth < MAX_DEPTH) {
            #pragma omp task default(none) shared(array, helper) firstprivate(k, current, p_end, depth)
            divide(array, helper, k, current, p_end, depth + 1);
        } else {
            divide(array, helper, k, current, p_end, depth + 1);
        }

        current = p_end + 1;
    }
    #pragma omp taskwait

    parallel_k_merge_ranks(array, helper, parts, start);

    std::copy(helper.begin() + start, helper.begin() + end + 1,
              array.begin() + start);
}

void parallel_k_way_mergesort_ranks(std::vector<int>& array, const int k) {
    if (array.size() <= 1) return;
    std::vector<int> helper(array.size());

    #pragma omp parallel
    #pragma omp single
    divide(array, helper, k, 0, (int)array.size() - 1, 0);
}