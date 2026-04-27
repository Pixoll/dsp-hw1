#include "k_way_mergesort.hpp"

#include <algorithm>
#include <limits>
#include <vector>

static void divide(std::vector<int> &array, std::vector<int> &helper, int k, int start, int end, int threshold);

static void merge(std::vector<int> &array, std::vector<int> &helper, int k, int low, int high);

void parallel_k_way_mergesort(std::vector<int> &array, const int k) {
    parallel_k_way_mergesort(array, k, 8192); // NOLINT(*-narrowing-conversions)
}

void parallel_k_way_mergesort(std::vector<int> &array, const int k, const int threshold) {
    std::vector helper(array);
    #pragma omp parallel default(none) shared(array, helper) firstprivate(k, threshold)
    #pragma omp single
    divide(array, helper, k, 0, array.size() - 1, threshold); // NOLINT(*-narrowing-conversions)
}

void divide(std::vector<int> &array, std::vector<int> &helper, const int k, int start, const int end, const int threshold) {
    const int size = end - start + 1;
    const int partitions_size = std::max(size / k, 1);
    const int last_partition_size = size - partitions_size * (k - 1);

    if (partitions_size > 1) {
        for (int i = 0; i < k; i++) {
            const int new_part_start = i * partitions_size + start;
            const int new_part_end = new_part_start - 1 + (i == k - 1 ? last_partition_size : partitions_size);
            if (partitions_size < threshold) {
                divide(array, helper, k, new_part_start, new_part_end, threshold);
            } else {
                #pragma omp task default(none) shared(array, helper) firstprivate(k, new_part_start, new_part_end, threshold)
                divide(array, helper, k, new_part_start, new_part_end, threshold);
            }
        }
        #pragma omp taskwait
    } else if (last_partition_size > 1) {
        start = end - last_partition_size;
    }

    merge(array, helper, k, start, end);
}

void merge(std::vector<int> &array, std::vector<int> &helper, int k, const int low, const int high) {
    const int size = high - low + 1;
    if (size < k) {
        k = size;
    }

    const int partitions_size = std::max(size / k, 1);
    const int last_partition_size = size - partitions_size * (k - 1);

    int indices[k];
    // omp parallel for degrades performance
    for (int i = 0; i < k; ++i) {
        indices[i] = 0;
    }

    int count = low;

    while (count <= high) {
        int min = std::numeric_limits<int>::max();
        int min_position = 0;
        int current_part = low;

        for (int i = 0; i < k; i++) {
            if (i == k - 1 && indices[i] == last_partition_size) {
                break;
            }

            if (indices[i] == partitions_size && i != k - 1) {
                current_part += partitions_size;
                continue;
            }

            if (helper[current_part + indices[i]] < min) {
                min = helper[current_part + indices[i]];
                min_position = i;
            }

            current_part += partitions_size;
        }

        array[count++] = min;
        indices[min_position]++;
    }

    std::copy(array.begin() + low, array.begin() + high + 1, helper.begin() + low);
}
