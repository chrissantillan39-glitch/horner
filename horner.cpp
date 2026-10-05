#include <time.h>
#include <memory>
#include <iostream>
extern "C" {
#include <immintrin.h>
}

using namespace std;

typedef unsigned long long bench_t;

static bench_t before;
static bench_t after;

static inline bench_t cycles(void) {
    unsigned int hi, lo;
    __asm__ __volatile__ ("rdtsc\n\t":"=a" (lo), "=d"(hi));
    return ((bench_t) lo) | (((bench_t) hi) << 32);
}

// ------------------------------------------------------------
// Horner escalar para float (correcto)
// coef[0] = término constante, coef[size-1] = mayor grado
// ------------------------------------------------------------
float horner_scalar(float X, const float *coef, int size) {
    float acc = 0.0f;
    for (int i = size - 1; i >= 0; --i) {
        acc = acc * X + coef[i];
    }
    return acc;
}

// ------------------------------------------------------------
// Horner SIMD con AVX (float) para 8 valores de X en paralelo
// X: __m256 con 8 valores de X
// coef: coeficientes del polinomio (mismos para las 8 evaluaciones)
// Retorna __m256 con los 8 resultados
// ------------------------------------------------------------
__m256 horner_simd(__m256 X, const float *coef, int size) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = size - 1; i >= 0; --i) {
        __m256 c = _mm256_set1_ps(coef[i]);
        acc = _mm256_add_ps(_mm256_mul_ps(acc, X), c);
    }
    return acc;
}

int main() {
    const int SIZE = 10000;
    const int NUM_TRIALS = 100000;
    const float X_val = 1.1f;

    // Alineación para AVX
    alignas(32) float coef[SIZE];
    alignas(32) float X_arr[8];

    // Inicializar coeficientes aleatorios
    srand(time(NULL));
    for (int i = 0; i < SIZE; i++) {
        coef[i] = (float)(rand() % 1000) / 1000.0f;
    }

    // Llenar los 8 valores de X (todos iguales para comparar)
    for (int i = 0; i < 8; i++) {
        X_arr[i] = X_val;
    }

    // Cargar X en un registro AVX
    __m256 X_vec = _mm256_load_ps(X_arr);

    // --------------------------------------------------------
    // Prueba de correctitud
    // --------------------------------------------------------
    float res_scalar = horner_scalar(X_val, coef, SIZE);
    __m256 res_simd_vec = horner_simd(X_vec, coef, SIZE);
    alignas(32) float res_simd[8];
    _mm256_store_ps(res_simd, res_simd_vec);

    cout << "Scalar: " << res_scalar << endl;
    cout << "SIMD:   " << res_simd[0] << " (deberían coincidir)" << endl;

    // --------------------------------------------------------
    // Benchmark escalar
    // --------------------------------------------------------
    clock_t t1 = clock();
    for (int j = 0; j < NUM_TRIALS; j++) {
        volatile float r = horner_scalar(X_val, coef, SIZE);
    }
    clock_t t2 = clock();
    double diff_scalar = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    cout << "Tiempo escalar: " << diff_scalar << " s" << endl;

    // --------------------------------------------------------
    // Benchmark SIMD
    // --------------------------------------------------------
    t1 = clock();
    for (int j = 0; j < NUM_TRIALS; j++) {
        __m256 r = horner_simd(X_vec, coef, SIZE);
        // Evitar optimización excesiva
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, r);
    }
    t2 = clock();
    double diff_simd = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    cout << "Tiempo SIMD:    " << diff_simd << " s" << endl;

    return 0;
}
