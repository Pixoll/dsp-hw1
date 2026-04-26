#include "k_way_mergesort.hpp"

#include <limits>
#include <vector>

#include "../util.hpp"

static void divide(std::vector<int> &array, std::vector<int> &helper, int k, int start, int end);

static void merge(std::vector<int> &array, std::vector<int> &helper, int k, int low, int high);

void sequential_k_way_mergesort(std::vector<int> &array, const int k) {
    std::vector helper(array);
    divide(array, helper, k, 0, array.size() - 1); // NOLINT(*-narrowing-conversions)
}

void divide(std::vector<int> &array, std::vector<int> &helper, const int k, const int start, const int end) {
    const int size = end - start + 1;
    const int partitions_size = MAX(size / k, 1);
    const int last_partition_size = size - partitions_size * (k - 1);

    if (partitions_size > 1) {
        for (int i = 0; i < k; i++) {
            const int new_part_start = i * partitions_size + start;
            const int new_part_end = new_part_start - 1 + (i == k - 1 ? last_partition_size : partitions_size);
            divide(array, helper, k, new_part_start, new_part_end);
        }
    } else if (last_partition_size > 1) {
        divide(array, helper, k, end - last_partition_size, end);
    }

    merge(array, helper, k, start, end);
}

void merge(std::vector<int> &array, std::vector<int> &helper, int k, const int low, const int high) {
    const int size = high - low + 1;
    if (size < k) {
        k = size;
    }

    const int partitions_size = MAX(size / k, 1);
    const int last_partition_size = size - partitions_size * (k - 1);

    int indices[k];
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

    for (int i = low; i <= high; i++) {
        helper[i] = array[i];
    }
}
