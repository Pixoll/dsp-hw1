#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#include "util.hpp"
#include "parallel/k_way_mergesort.hpp"
#include "sequential/k_way_mergesort.hpp"
#include "parallel/parallel_mergesort_ranks.hpp"

template<typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
    if (!v.empty()) {
        out << '[';
        for (int i = 0; i < v.size(); ++i) {
            out << v[i] << ", ";
        }
        out << "\b\b]";
    }
    return out;
}

int main() {
    std::vector<int> array1(1 << 20);
    int_generator<int> generate_int(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    for (int &n: array1) {
        n = generate_int();
    }

    std::vector array2(array1);
    std::vector array3(array1);

    std::vector array4(array1);

    std::cout << "sorting control" << std::endl;
    std::ranges::sort(array1);

    std::cout << "sorting sequential_k_way_mergesort" << std::endl;
    sequential_k_way_mergesort(array2, 4);

    std::cout << "sorting parallel_k_way_mergesort" << std::endl;
    parallel_k_way_mergesort(array3, 4);

    std::cout << "sorting parallel_mergesort_ranks (2-way with binary search ranks)" << std::endl;
    parallel_mergesort_ranks(array4);

    std::cout << "checking..." << std::endl;
    const bool array2_correct = array2 == array1;
    const bool array3_correct = array3 == array1;
    const bool array4_correct = array4 == array1;

    std::cout << std::boolalpha
        << "sequential k-way sort:  " << array2_correct << '\n'
        << "parallel k-way sort:    " << array3_correct << '\n'
        << "parallel ranks sort:     " << array4_correct << std::endl;

    return 0;
}