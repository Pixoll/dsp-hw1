#include "mergesort_paralelo.h"
#include "../secuencial/mergesort.h"
#include <omp.h>

void mergesort_paralelo(int* A, int* temp, int left, int right, int umbral){
    if(left < right){
        if (right - left < umbral) {
            mergesort_secuencial(A, temp, left, right);
        } else {
            int mid = left + (right - left)/2;

            // Se crean tareas para ordenar cada mitad en paralelo
            #pragma omp task shared(A, temp) firstprivate(left, mid, umbral) // primera mitad
            mergesort_paralelo(A, temp, left, mid, umbral);

            #pragma omp task shared(A, temp) firstprivate(mid, right, umbral) // segunda mitad
            mergesort_paralelo(A, temp, mid + 1, right, umbral);

            #pragma omp taskwait // Espera a que ambas tareas terminen
            merge(A, temp, left, mid, right);
        }
    }
}

void iniciar_mergesort_paralelo(int* A, int* temp, int n, int umbral){
    #pragma omp parallel // equipo de hebras
    {
        #pragma omp single // hebra inicial hace primera llamada recursiva
        {
            mergesort_paralelo(A, temp, 0, n - 1, umbral);
        }
    }
}