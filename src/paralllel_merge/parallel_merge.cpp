#include <vector>
#include <algorithm>
#include <omp.h>


static int binary_search_rank(int x, const std::vector<int> &A, int left, int right) {
    int low = left;
    int high = std::max(left, right + 1);
    while (low < high) {
        if (const int mid = low + (high - low) / 2; A[mid] < x) low = mid + 1;
        else high = mid;
    }
    return low;
}


static void sequential_merge(const std::vector<int> &src, std::vector<int> &dest, int l1, int r1, int l2, int r2, int out_pos) {
    int i = l1, j = l2, k = out_pos;
    while (i <= r1 && j <= r2) {
        if (src[i] <= src[j]) dest[k++] = src[i++];
        else dest[k++] = src[j++];
    }
    while (i <= r1) dest[k++] = src[i++];
    while (j <= r2) dest[k++] = src[j++];
}


void parallel_merge_ranks(const std::vector<int> &src, std::vector<int> &dest, int l1, int r1, int l2, int r2, int out_pos) {

    int n1 = r1 - l1 + 1;
    int n2 = r2 - l2 + 1;

    static constexpr int MERGE_THRESHOLD = 2048;
    if (n1 + n2 < MERGE_THRESHOLD) {
        sequential_merge(src, dest, l1, r1, l2, r2, out_pos);
        return;
    }

    if (n1 < n2) {
        std::swap(l1, l2); std::swap(r1, r2); std::swap(n1, n2);
    }

    if (n1 <= 0) return;

    int mid1 = l1 + n1 / 2;
    int x = src[mid1];

    int mid2 = binary_search_rank(x, src, l2, r2);

    int pos = out_pos + (mid1 - l1) + (mid2 - l2);
    dest[pos] = x;

    #pragma omp task default(none) shared(src, dest) firstprivate(l1, mid1, l2, mid2, out_pos)
    parallel_merge_ranks(src, dest, l1, mid1 - 1, l2, mid2 - 1, out_pos);

    #pragma omp task default(none) shared(src, dest) firstprivate(mid1, r1, mid2, r2, pos)
    parallel_merge_ranks(src, dest, mid1 + 1, r1, mid2, r2, pos + 1);
}