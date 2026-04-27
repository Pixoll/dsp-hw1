#include "ranks_k_way_mergesort.hpp"

#include <algorithm>
#include <vector>

#include "../util.hpp"
#include "merge/ranks_k_way.hpp"

static void divide(std::vector<int> &array, std::vector<int> &helper, int k, int left, int right, int g_threshold);

static void merge(
    std::vector<int> &array,
    std::vector<int> &helper,
    int left,
    int right,
    const std::vector<Range> &partitions,
    int g_threshold
);

void parallel_ranks_k_way_mergesort(std::vector<int> &array, const int k, const int g_threshold) {
    std::vector<int> helper(array.size());

    #pragma omp parallel default(none) shared(array, helper) firstprivate(k, g_threshold)
    #pragma omp single
    divide(array, helper, k, 0, array.size() - 1, g_threshold); // NOLINT(*-narrowing-conversions)
}

void divide(
    std::vector<int> &array,
    std::vector<int> &helper,
    const int k,
    const int left,
    const int right,
    const int g_threshold
) {
    if (right - left < g_threshold) {
        std::sort(array.begin() + left, array.begin() + right + 1);
        std::copy(array.begin() + left, array.begin() + right + 1, helper.begin() + left);
        return;
    }

    std::vector<Range> partitions;
    int current = left;

    for (int i = 0; i < k && current <= right; i++) {
        const int p_size = (right - current + 1) / (k - i);
        const int p_end = current + p_size - 1;
        partitions.emplace_back(current, p_end);

        #pragma omp task default(none) shared(array, helper) firstprivate(k, current, p_end, g_threshold)
        divide(array, helper, k, current, p_end, g_threshold);

        current = p_end + 1;
    }
    #pragma omp taskwait

    merge(array, helper, left, right, partitions, g_threshold);
}

void merge(
    std::vector<int> &array,
    std::vector<int> &helper,
    const int left,
    const int right,
    const std::vector<Range> &partitions,
    const int g_threshold
) {
    #pragma omp taskgroup
    {
        parallel_ranks_k_way_merge(array, helper, partitions, left, g_threshold);
    }

    std::copy(helper.begin() + left, helper.begin() + right + 1, array.begin() + left);
}
