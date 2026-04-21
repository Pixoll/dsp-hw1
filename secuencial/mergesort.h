#ifndef MERGESORT_H
#define MERGESORT_H

void merge(int* A, int* temp, int left, int mid, int right);
void mergesort_secuencial(int* A, int* temp, int left, int right);
bool esta_ordenado(int* A, int n);

#endif