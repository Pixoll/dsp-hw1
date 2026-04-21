#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

#include "secuencial/mergesort.h"
#include "paralelo/mergesort_paralelo.h"

int main() {
   int e = 20;
   int n = 1 << e; // 2^e 
   
   int umbral = 10000; // Umbral de corte: Arreglos < 10000 elementos se ordenan secuencialmente

    std::cout << "Numero de elementos: " << n << std::endl;

    std::vector<int> A(n);
    std::vector<int> A_original(n);

    std::mt19937 rng(123); // Semilla fija para reproducibilidad
    std::uniform_int_distribution<int> dist(1, 1000000); // Rango de numeros aleatorios

    for (int i = 0; i < n; i++) {
        A_original[i] = dist(rng);
    }

    // prueba secuencial: MergeSort

    std::cout << "-MergeSort Secuencial-" << std::endl;

    std::vector<int> A_sec = A_original;
    std::vector<int> temp_sec(n);

    double start_time = omp_get_wtime();
    mergesort_secuencial(A_sec.data(), temp_sec.data(), 0, n - 1);
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    
    // prueba paralelo: MergeSort

    std::vector<int> A_par = A_original;
    std::vector<int> temp_par(n); // Vector temporal para mergesort paralelo

    omp_set_num_threads(4); // hebras a experimentar {1, 2, 4, 8}

    double start_time_par = omp_get_wtime();
    iniciar_mergesort_paralelo(A_par.data(), temp_par.data(), n, umbral);
    double end_time_par = omp_get_wtime();
    double elapsed_time_par = end_time_par - start_time_par;

    std::cout << "Tiempo secuencial MergeSort: " << elapsed_time << " segundos" << std::endl;
    std::cout << "Tiempo paralelo MergeSort: " << elapsed_time_par << " segundos" << std::endl;

    // speeup = T1/Tp
    double speedup = elapsed_time / elapsed_time_par;
    std::cout << "Speedup: " << speedup << std::endl;

    //eficiencia (p) = speedup/p
    int p = 4; // numero de hebras usadas {1, 2, 4, 8}
    double efficiency = (speedup / p)*100;
    std::cout << "Eficiencia: " << efficiency << "%" << std::endl;

    return 0;
}
