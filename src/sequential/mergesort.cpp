#include "../sequential/mergesort.hpp"

static void divide(std::vector<int> &array, std::vector<int> &helper, int left, int right);

static void merge(std::vector<int> &array, std::vector<int> &helper, int left, int mid, int right);

void sequential_mergesort(std::vector<int> &array) {
    std::vector helper(array);
    divide(array, helper, 0, array.size() - 1); // NOLINT(*-narrowing-conversions)
}

void divide(std::vector<int> &array, std::vector<int> &helper, const int left, const int right) {
    if (left >= right) {
        return;
    }

    const int mid = left + (right - left) / 2;

    divide(array, helper, left, mid);
    divide(array, helper, mid + 1, right);

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

    for (i = left; i <= right; i++) {
        array[i] = helper[i];
    }
}
