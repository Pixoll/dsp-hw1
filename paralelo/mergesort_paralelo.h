#ifndef MERGESORT_PARALELO_H
#define MERGESORT_PARALELO_H

void mergesort_paralelo_recursivo(int* A, int* temp, int left, int right, int umbral);
void iniciar_mergesort_paralelo(int* A, int* temp, int n, int umbral);

#endif