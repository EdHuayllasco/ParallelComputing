#include <iostream>
#include <chrono>
using namespace std;

// Función para multiplicar matrices por bloques
void multiplyMatricesBlock(int **A, int **B, int **C, int N, int blockSize) {
    for (int i = 0; i < N; i += blockSize) {
        for (int j = 0; j < N; j += blockSize) {
            for (int k = 0; k < N; k += blockSize) {
                for (int ii = i; ii < min(i + blockSize, N); ++ii) {
                    for (int jj = j; jj < min(j + blockSize, N); ++jj) {
                        for (int kk = k; kk < min(k + blockSize, N); ++kk) {
                            C[ii][jj] += A[ii][kk] * B[kk][jj];
                        }
                    }
                }
            }
        }
    }
}

int main() {
    int tamanios[] = {10, 20, 30, 50, 100, 500, 1000, 2000};

    for (int t = 0; t < sizeof(tamanios) / sizeof(tamanios[0]); ++t) {
        int N = tamanios[t];
        int blockSize = N / 2;  // Tamaño del bloque es la mitad de N

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

        // Multiplicar las matrices utilizando la multiplicación por bloques
        multiplyMatricesBlock(A, B, C, N, blockSize);

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