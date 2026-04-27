#include "mergesort.hpp"

static void divide(std::vector<int> &array, std::vector<int> &helper, int left, int right);
static void merge(std::vector<int> &array, std::vector<int> &helper, int left, int mid, int right);

static constexpr int TASK_THRESHOLD = 8192;

void parallel_mergesort(std::vector<int> &array) {
    std::vector helper(array);
    #pragma omp parallel default(none) shared(array, helper)
    #pragma omp single
    divide(array, helper, 0, array.size() - 1); // NOLINT(*-narrowing-conversions)
}

void divide(std::vector<int> &array, std::vector<int> &helper, const int left, const int right) {
    if (left >= right) {
        return;
    }

    const int size = right - left;
    const int mid = left + size / 2;

    if (size < TASK_THRESHOLD) {
        divide(array, helper, left, mid);
        divide(array, helper, mid + 1, right);
    } else {
        #pragma omp task default(none) shared(array, helper) firstprivate(left, mid)
        divide(array, helper, left, mid);
        #pragma omp task default(none) shared(array, helper) firstprivate(mid, right)
        divide(array, helper, mid + 1, right);
        #pragma omp taskwait
    }

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
