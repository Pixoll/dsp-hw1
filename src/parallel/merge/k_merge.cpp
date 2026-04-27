#include "k_merge.hpp"

#include <algorithm>
#include <limits>
#include <vector>

struct MaxPivot {
    int max_size = -1;
    int pivot_list_idx = -1;
};

#pragma omp declare \
reduction(maxpivot : MaxPivot : omp_out = (omp_in.max_size > omp_out.max_size) ? omp_in : omp_out) \
initializer(omp_priv = {-1, -1})

static constexpr int K_MERGE_THRESHOLD = 16384;

static int binary_search_rank(int x, const std::vector<int> &array, int left, int right);

static void sequential_k_merge(
    const std::vector<int> &array,
    std::vector<int> &helper,
    const std::vector<Range> &partitions,
    int out_pos
);

void parallel_k_merge_ranks(
    const std::vector<int> &array,
    std::vector<int> &helper,
    const std::vector<Range> &partitions,
    const int start
) {
    const int k = partitions.size(); // NOLINT(*-narrowing-conversions)
    int total_elements = 0;
    MaxPivot max_pivot{};

    #pragma omp parallel for default(none) \
            shared(partitions, k) \
            reduction(+:total_elements) \
            reduction(maxpivot:max_pivot)
    for (int i = 0; i < k; ++i) {
        const int s = partitions[i].size();
        total_elements += s;

        if (s > max_pivot.max_size) {
            max_pivot.max_size = s;
            max_pivot.pivot_list_idx = i;
        }
    }

    if (total_elements < K_MERGE_THRESHOLD) {
        sequential_k_merge(array, helper, partitions, start);
        return;
    }

    const int pivot_list_idx = max_pivot.pivot_list_idx;
    const int mid_in_list = partitions[pivot_list_idx].start + max_pivot.max_size / 2;
    const int x = array[mid_in_list];

    std::vector<int> split_indices(k);
    int global_rank = 0;

    #pragma omp parallel for default(none) shared(k, x, array, partitions, split_indices) reduction(+:global_rank)
    for (int i = 0; i < k; ++i) {
        const int res = binary_search_rank(x, array, partitions[i].start, partitions[i].end);
        split_indices[i] = res;
        global_rank += res - partitions[i].start;
    }

    const int final_pos = start + global_rank;
    helper[final_pos] = x;

    std::vector<Range> left_parts(k);
    std::vector<Range> right_parts(k);

    #pragma omp parallel for default(none) shared(k, partitions, left_parts, right_parts, split_indices, pivot_list_idx)
    for (int i = 0; i < k; ++i) {
        left_parts[i].start = partitions[i].start;
        left_parts[i].end = split_indices[i] - 1;

        right_parts[i].start = split_indices[i] + (i == pivot_list_idx);
        right_parts[i].end = partitions[i].end;
    }

    #pragma omp task default(none) shared(array, helper) firstprivate(start, left_parts)
    parallel_k_merge_ranks(array, helper, left_parts, start);

    #pragma omp task default(none) shared(array, helper) firstprivate(final_pos, right_parts)
    parallel_k_merge_ranks(array, helper, right_parts, final_pos + 1);
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

void sequential_k_merge(
    const std::vector<int> &array,
    std::vector<int> &helper,
    const std::vector<Range> &partitions,
    const int out_pos
) {
    const int k = partitions.size(); // NOLINT(*-narrowing-conversions)
    std::vector<int> current_indices(k);
    for (int i = 0; i < k; ++i) {
        current_indices[i] = partitions[i].start;
    }

    int total_elements = 0;
    for (const auto &p: partitions) {
        total_elements += p.size();
    }

    for (int count = 0; count < total_elements; ++count) {
        int min_val = std::numeric_limits<int>::max();
        int min_idx = -1;

        for (int i = 0; i < k; ++i) {
            const int ci = current_indices[i];
            if (ci <= partitions[i].end && array[ci] < min_val) {
                min_val = array[ci];
                min_idx = i;
            }
        }

        helper[out_pos + count] = min_val;
        current_indices[min_idx]++;
    }
}
