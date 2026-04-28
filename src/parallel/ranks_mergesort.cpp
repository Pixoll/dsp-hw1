#include "ranks_mergesort.hpp"

#include <algorithm>

#include "merge/ranks.hpp"

static void divide(std::vector<int> &array, std::vector<int> &helper, int left, int right, int g_threshold);

static void merge(std::vector<int> &array, std::vector<int> &helper, int left, int mid, int right, int g_threshold);

void parallel_ranks_mergesort(std::vector<int> &array, const int g_threshold) {
    std::vector helper(array);

    #pragma omp parallel default(none) shared(array, helper) firstprivate(g_threshold)
    #pragma omp single
    divide(array, helper, 0, array.size() - 1, g_threshold); // NOLINT(*-narrowing-conversions)
}

void divide(std::vector<int> &array, std::vector<int> &helper, const int left, const int right, const int g_threshold) {
    if (left >= right) {
        return;
    }

    const int mid = left + (right - left) / 2;

    if (right - left < g_threshold) {
        divide(array, helper, left, mid, g_threshold);
        divide(array, helper, mid + 1, right, g_threshold);
    } else {
        #pragma omp task default(none) shared(array, helper) firstprivate(left, mid, g_threshold)
        divide(array, helper, left, mid, g_threshold);
        #pragma omp task default(none) shared(array, helper) firstprivate(mid, right, g_threshold)
        divide(array, helper, mid + 1, right, g_threshold);
        #pragma omp taskwait
    }

    merge(array, helper, left, mid, right, g_threshold);
}

void merge(
    std::vector<int> &array,
    std::vector<int> &helper,
    const int left,
    const int mid,
    const int right,
    const int g_threshold
) {
    if (right - left < g_threshold) {
        sequential_merge(array, helper, left, mid, mid + 1, right, left);
    } else {
        #pragma omp taskgroup
        parallel_ranks_merge(array, helper, left, mid, mid + 1, right, left, g_threshold);
    }

    std::copy(helper.begin() + left, helper.begin() + right + 1, array.begin() + left);
}
