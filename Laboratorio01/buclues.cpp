#include <iostream>
#include <ctime>
#include <cstdlib>
#include <chrono>
using namespace std;

void inicializar_matriz(double **&A, int MAX){
    A = new double*[MAX];
    for( int i=0; i<MAX; i++ )
        A[i] = new double[MAX];
}

void llenar_matriz(double **A, int MAX){
    srand(time(NULL));
    for (int i=0;i<MAX;i++)
        for (int j=0;j<MAX;j++)
            A[i][j] = 1 + rand() % (11 - 1);
}

void llenar_vector(double *x, int MAX){
    for (int i=0;i<MAX;i++)
        x[i] = 1 + rand() % (11 - 1);
}

void llenar_matriz_con_ceros(double **A, int MAX){
    for (int i=0;i<MAX;i++)
        for(int j=0;j<MAX;j++)
            A[i][j] = 0;
}

void llenar_vector_con_ceros(double *y, int MAX){
    for (int i=0;i<MAX;i++)
        y[i] = 0;
}

int main(){
    int MAX = 50000;
    double **A;
    double *x = new double[MAX]; 
    double *y = new double[MAX];
    // Inicialización de matriz
    inicializar_matriz(A,MAX);
    // Llenado de datos
    llenar_matriz(A,MAX);
    llenar_vector(x,MAX);

    // COMPARACIÓN
    // Bucle 1
    llenar_vector_con_ceros(y,MAX);
    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < MAX; i++)
        for (int j = 0; j < MAX; j++)
            y[i] += A[i][j]*x[j];
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float,std::milli> duration1 = end1 - start1;
    cout<<"Tiempo bucle 1: "<<duration1.count()<<endl<<endl;

    // Bucle 2
    llenar_vector_con_ceros(y,MAX);
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < MAX; j++)
        for (int i = 0; i < MAX; i++)
            y[i] += A[i][j]*x[j];
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float,std::milli> duration2 = end2 - start2;
    cout<<"Tiempo bucle 2: "<<duration2.count()<<endl<<endl;
    
    return 0;
}
