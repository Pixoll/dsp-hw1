#include "ranks_k_way_mergesort.hpp"

#include <algorithm>
#include <vector>

#include "../util.hpp"
#include "merge/ranks_k_way.hpp"

static constexpr int SORT_THRESHOLD = 4096;
static constexpr int MAX_DEPTH = 4;

static void divide(std::vector<int> &array, std::vector<int> &helper, int k, int start, int end, int depth);

static void merge(
    std::vector<int> &array,
    std::vector<int> &helper,
    int left,
    int right,
    const std::vector<Range> &partitions
);

void parallel_ranks_k_way_mergesort(std::vector<int> &array, const int k) {
    if (array.size() <= 1) return;

    std::vector<int> helper(array.size());

    #pragma omp parallel default(none) shared(array, helper) firstprivate(k)
    #pragma omp single
    divide(array, helper, k, 0, array.size() - 1, 0); // NOLINT(*-narrowing-conversions)
}

void divide(
    std::vector<int> &array,
    std::vector<int> &helper,
    const int k,
    const int start,
    const int end,
    const int depth
) {
    const int size = end - start + 1;
    if (size <= 1) return;

    if (size < SORT_THRESHOLD) {
        std::sort(array.begin() + start, array.begin() + end + 1);
        return;
    }

    std::vector<Range> partitions;
    int current = start;

    for (int i = 0; i < k && current <= end; i++) {
        const int p_size = (end - current + 1) / (k - i);
        const int p_end = current + p_size - 1;
        partitions.emplace_back(current, p_end);

        if (depth < MAX_DEPTH) {
            #pragma omp task default(none) shared(array, helper) firstprivate(k, current, p_end, depth)
            divide(array, helper, k, current, p_end, depth + 1);
        } else {
            std::sort(array.begin() + current, array.begin() + p_end + 1);
        }

        current = p_end + 1;
    }
    #pragma omp taskwait

    merge(array, helper, start, end, partitions);
}

void merge(
    std::vector<int> &array,
    std::vector<int> &helper,
    const int left,
    const int right,
    const std::vector<Range> &partitions
) {
    #pragma omp taskgroup
    {
        parallel_ranks_k_way_merge(array, helper, partitions, left);
    }

    const int total_size = right - left + 1;
    if (total_size > 100000) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < total_size; ++i) {
            array[left + i] = helper[left + i];
        }
    } else {
        std::copy(helper.begin() + left, helper.begin() + right + 1, array.begin() + left);
    }
}
