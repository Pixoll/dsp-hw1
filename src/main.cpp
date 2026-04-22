#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>

#include "parallel/k_way_mergesort.hpp"
#include "sequential/k_way_mergesort.hpp"

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
    const std::vector array{
        32, 15, 8, 23, 4, 42, 16, 9, 3, 27, 11, 19, 2, 31, 7, 14,
        6, 25, 18, 1, 10, 22, 5, 13, 20, 28, 12, 17, 21, 26, 24, 30,
    };
    std::vector array1(array);
    std::vector array2(array);
    std::vector array3(array);

    sequential_k_way_mergesort(array1, 4);
    parallel_k_way_mergesort(array2, 4);
    std::ranges::sort(array3);

    std::cout
        << "original:              " << array << '\n'
        << "sequential k-way sort: " << array1 << '\n'
        << "parallel k-way sort:   " << array1 << '\n'
        << "normal sort:           " << array3 << std::endl;

    return 0;
}
