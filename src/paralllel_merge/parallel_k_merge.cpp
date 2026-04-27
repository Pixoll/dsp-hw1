#include <vector>
#include <algorithm>
#include <omp.h>

struct Range {
    int start;
    int end;
    int size() const { return (end >= start) ? (end - start + 1) : 0; }
};


static int binary_search_rank(int x, const std::vector<int> &A, int left, int right) {
    int low = left;
    int high = std::max(left, right + 1);
    while (low < high) {
        int mid = low + (high - low) / 2;
        if (A[mid] < x) low = mid + 1;
        else high = mid;
    }
    return low;
}

static void sequential_k_merge(const std::vector<int> &src, std::vector<int> &dest,
                               const std::vector<Range> &partitions, int out_pos) {
    int k = partitions.size();
    std::vector<int> current_indices(k);
    for(int i = 0; i < k; ++i) current_indices[i] = partitions[i].start;

    int total_elements = 0;
    for(const auto& p : partitions) total_elements += p.size();

    for (int count = 0; count < total_elements; ++count) {
        int min_val = 2147483647;
        int min_idx = -1;

        for (int i = 0; i < k; ++i) {
            if (current_indices[i] <= partitions[i].end) {
                if (src[current_indices[i]] < min_val) {
                    min_val = src[current_indices[i]];
                    min_idx = i;
                }
            }
        }
        dest[out_pos + count] = min_val;
        current_indices[min_idx]++;
    }
}

void parallel_k_merge_ranks(const std::vector<int> &src, std::vector<int> &dest,
                            std::vector<Range> partitions, int out_pos) {

    int total_elements = 0;
    int max_size = -1;
    int pivot_list_idx = -1;

    for (int i = 0; i < (int)partitions.size(); ++i) {
        int s = partitions[i].size();
        total_elements += s;
        if (s > max_size) {
            max_size = s;
            pivot_list_idx = i;
        }
    }

    static constexpr int K_MERGE_THRESHOLD = 16384;
    if (total_elements < K_MERGE_THRESHOLD) {
        sequential_k_merge(src, dest, partitions, out_pos);
        return;
    }

    int mid_in_list = partitions[pivot_list_idx].start + (max_size / 2);
    int x = src[mid_in_list];

    std::vector<int> split_indices(partitions.size());
    int global_rank = 0;

    for (int i = 0; i < (int)partitions.size(); ++i) {
        split_indices[i] = binary_search_rank(x, src, partitions[i].start, partitions[i].end);
        global_rank += (split_indices[i] - partitions[i].start);
    }

    int final_pos = out_pos + global_rank;
    dest[final_pos] = x;

    std::vector<Range> left_parts(partitions.size());
    std::vector<Range> right_parts(partitions.size());

    for (int i = 0; i < (int)partitions.size(); ++i) {
        left_parts[i] = {partitions[i].start, split_indices[i] - 1};

        if (i == pivot_list_idx) {
            right_parts[i] = {split_indices[i] + 1, partitions[i].end};
        } else {
            right_parts[i] = {split_indices[i], partitions[i].end};
        }
    }

    #pragma omp task default(none) shared(src, dest) firstprivate(out_pos, left_parts)
    parallel_k_merge_ranks(src, dest, left_parts, out_pos);

    #pragma omp task default(none) shared(src, dest) firstprivate(final_pos, right_parts)
    parallel_k_merge_ranks(src, dest, right_parts, final_pos + 1);
}