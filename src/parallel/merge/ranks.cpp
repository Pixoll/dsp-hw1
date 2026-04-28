#include "ranks.hpp"

#include <algorithm>
#include <vector>

static int binary_search_rank(int x, const std::vector<int> &array, int left, int right);

void parallel_ranks_merge(
    const std::vector<int> &array,
    std::vector<int> &helper,
    int l1,
    int r1,
    int l2,
    int r2,
    const int start,
    const int g_threshold
) {
    int n1 = r1 - l1 + 1;
    int n2 = r2 - l2 + 1;

    if (n1 + n2 < g_threshold) {
        sequential_merge(array, helper, l1, r1, l2, r2, start);
        return;
    }

    if (n1 < n2) {
        std::swap(l1, l2);
        std::swap(r1, r2);
        std::swap(n1, n2);
    }

    if (n1 <= 0) {
        return;
    }

    const int mid1 = l1 + n1 / 2;
    const int x = array[mid1];
    const int mid2 = binary_search_rank(x, array, l2, r2);
    const int pos = start + (mid1 - l1) + (mid2 - l2);
    helper[pos] = x;

    #pragma omp task default(none) shared(array, helper) firstprivate(l1, mid1, l2, mid2, start, g_threshold)
    parallel_ranks_merge(array, helper, l1, mid1 - 1, l2, mid2 - 1, start, g_threshold);

    #pragma omp task default(none) shared(array, helper) firstprivate(mid1, r1, mid2, r2, pos, g_threshold)
    parallel_ranks_merge(array, helper, mid1 + 1, r1, mid2, r2, pos + 1, g_threshold);
}

int binary_search_rank(const int x, const std::vector<int> &array, const int left, const int right) {
    int low = left;
    int high = std::max(left, right + 1);

    while (low < high) {
        const int mid = low + (high - low) / 2;
        if (array[mid] < x) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    return low;
}

void sequential_merge(
    const std::vector<int> &array,
    std::vector<int> &helper,
    const int l1,
    const int r1,
    const int l2,
    const int r2,
    const int out_pos
) {
    int i = l1;
    int j = l2;
    int k = out_pos;

    while (i <= r1 && j <= r2) {
        if (array[i] <= array[j]) {
            helper[k++] = array[i++];
        } else {
            helper[k++] = array[j++];
        }
    }

    while (i <= r1) {
        helper[k++] = array[i++];
    }
    while (j <= r2) {
        helper[k++] = array[j++];
    }
}
