#include <iostream>
#include <chrono>
#include <cstdlib>
using namespace std;

// Función para multiplicar matrices de forma clásica con punteros
void multiplyMatrices(int **A, int **B, int **C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    int tamanios[] = {10, 20, 30, 50, 100, 500, 1000, 10000};

    for (int i = 0; i < sizeof(tamanios) / sizeof(tamanios[0]); ++i) {
        int N = tamanios[i];

        // Crear matrices utilizando punteros
        int **A, **B, **C;
        A = new int*[N];
        B = new int*[N];
        C = new int*[N];

        for (int i = 0; i < N; ++i) {
            A[i] = new int[N];
            B[i] = new int[N];
            C[i] = new int[N];
        }

        // Llenar las matrices A y B con valores aleatorios (para simplificar, asumimos valores aleatorios)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                A[i][j] = rand() % 100;
                B[i][j] = rand() % 100;
            }
        }

        auto start = chrono::high_resolution_clock::now();

        // Multiplicar las matrices
        multiplyMatrices(A, B, C, N);

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

        cout << "Tiempo de ejecución para N = " << N << ": " << duration.count() << " milisegundos" << endl;

        // Liberar la memoria asignada para esta matriz
        for (int i = 0; i < N; ++i) {
            delete[] A[i];
            delete[] B[i];
            delete[] C[i];
        }

        delete[] A;
        delete[] B;
        delete[] C;
    }

    return 0;
}