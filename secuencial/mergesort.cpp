#include "mergesort.h"

void merge(int* A, int* temp, int left, int mid, int right){
    int i = left;
    int j = mid + 1;
    int k = left;

    while (i <= mid && j <= right){
        if(A[i] <= A[j]){
            temp[k++] = A[i++];
        } else {
            temp[k++] = A[j++];
        }
    }

    while (i <= mid){
        temp[k++] = A[i++];
    }

    while (j <= right){
        temp[k++] = A[j++];
    }

    for (i = left; i <= right; i++){
        A[i] = temp[i];
    }
}

void mergesort_secuencial(int* A, int* temp, int left, int right){
    if(left < right){
        int mid = left + (right - left)/2;

        // Se ordena recursivamente cada mitad
        mergesort_secuencial(A, temp, left, mid);
        mergesort_secuencial(A, temp, mid + 1, right);

        merge(A, temp, left, mid, right);
    }
}

bool esta_ordenado(int* A, int n){
    for (int i = 0; i < n-1; i++){
        if (A[i] > A[i+1]){
            return false;
        }
    }
    return true;
}