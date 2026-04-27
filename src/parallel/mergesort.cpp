#include "mergesort.hpp"
#include <algorithm>

static void divide(std::vector<int> &array, std::vector<int> &helper, int left, int right, int threshold);
static void merge(std::vector<int> &array, std::vector<int> &helper, int left, int mid, int right);

void parallel_mergesort(std::vector<int> &array) {
    parallel_mergesort(array, 8192); // NOLINT(*-narrowing-conversions)
}

void parallel_mergesort(std::vector<int> &array, int threshold) {
    std::vector helper(array);
    #pragma omp parallel default(none) shared(array, helper, threshold)
    #pragma omp single
    divide(array, helper, 0, array.size() - 1, threshold); // NOLINT(*-narrowing-conversions)
}

void divide(std::vector<int> &array, std::vector<int> &helper, const int left, const int right, const int threshold) {
    if (left >= right) {
        return;
    }

    const int size = right - left;

    if (size < threshold) {
       std::sort(array.begin() + left, array.begin() + right + 1);
       return;
    }

    const int mid = left + (right - left) / 2;

    #pragma omp task default(none) shared(array, helper) firstprivate(left, mid, threshold)
    divide(array, helper, left, mid, threshold);

    #pragma omp task default(none) shared(array, helper) firstprivate(mid, right, threshold)
    divide(array, helper, mid + 1, right, threshold);

    #pragma omp taskwait

    merge(array, helper, left, mid, right);
}

void merge(std::vector<int> &array, std::vector<int> &helper, const int left, const int mid, const int right) {
    int i = left;
    int j = mid + 1;
    int k = left;

    while (i <= mid && j <= right) {
        if (array[i] <= array[j]) {
            helper[k++] = array[i++];
        } else {
            helper[k++] = array[j++];
        }
    }

    while (i <= mid) {
        helper[k++] = array[i++];
    }

    while (j <= right) {
        helper[k++] = array[j++];
    }

    std::copy(helper.begin() + left, helper.begin() + right + 1, array.begin() + left);
}
