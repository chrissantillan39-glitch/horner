#include <time.h>
#include <memory>
#include <iostream>
extern "C"
{
#include <immintrin.h>
}

using namespace std;

// COUNTER TYPE //
typedef unsigned long long bench_t;

static bench_t before;
static bench_t after;

// ASKS FOR THE TIME STAMP COUNTER (= WHAT TIME IS IT PLEASE ?)//
static inline bench_t cycles(void) {
	unsigned int hi, lo;
	__asm__ __volatile__ ("rdtsc\n\t":"=a" (lo), "=d"(hi));
	return ((bench_t) lo) | (((bench_t) hi) << 32);
}

// Función escalar adaptada a float para comparar
float horner(float X, float *coef, long size){
	float ACC = 0.0f;
	int i;
	for(i = 0; i < size; i++){
		ACC = (ACC + coef[i]) * X;
	}
	return ACC;	
}

// Función AVX intrínseca modificada para float
float horner_intrinsic(float X, float *coef, long size){
	float *R, P;
	int i;
	__m256 *ymm0, X256, Y;	
	
	ymm0 = (__m256*)coef; 
	
	// Precalculamos las potencias de X necesarias
	float X2 = X*X;
	float X3 = X2*X;
	float X4 = X3*X;
	float X5 = X4*X;
	float X6 = X5*X;
	float X7 = X6*X;
	float X8 = X7*X;
	
	// Usamos _ps (packed single) en lugar de _pd (packed double)
	X256 = _mm256_set1_ps(X8);
	Y = _mm256_set1_ps(0.0f);
    
    // Un registro de 256 bits ahora procesa 8 floats por iteración
	for(i = 0; i < size/8 - 1; i++){
		Y = _mm256_add_ps(Y, ymm0[i]);
		Y = _mm256_mul_ps(Y, X256);
	}
	
	Y = _mm256_add_ps(Y, ymm0[i]);
	
	R = (float *)(&Y);
	
	// La reconstrucción final debe abarcar las 8 posiciones (R[0] a R[7])
	P  = R[7] * X;
	P += R[6] * X2;
	P += R[5] * X3;
	P += R[4] * X4;
	P += R[3] * X5;
	P += R[2] * X6;
	P += R[1] * X7;
	P += R[0] * X8;

	return P;
}

int main(){
	alignas(32) float coef[4000]; // Cambiado a float (aunque no se usa abajo)
	float X = 1.1f;
	float R;
	int i, num_trails = 100000;
	
	clock_t t1, t2;
	
	srand(time(NULL));	

	float *coeficientes;
	int j;
	// Alojamiento alineado a 32 bytes para floats
	coeficientes = (float *)_mm_malloc(10000 * sizeof(float), 32);

	for(j = 0; j < 1; j++){
		for(i = 0; i < 10000; i++){
			coeficientes[i] = (float)(rand()%1000)/1000.0f;
			if(i < 10)
				cout << coeficientes[i] << endl;
		}
		R = horner(X, coeficientes, 10000);
		cout << "Escalar: " << R << endl;
		R = horner_intrinsic(X, coeficientes, 10000);
		cout << "Intrinsics: " << R << endl;		
	}
	
	t1 = clock();
	for(j = 0; j < num_trails; j++)	
		R = horner(X, coeficientes, 10000);
	t2 = clock();
  	float diff_esc = (((float)t2 - (float)t1) / CLOCKS_PER_SEC );
  	cout << "Time taken (Escalar): " << diff_esc << endl;
	
	t1 = clock();
	for(j = 0; j < num_trails; j++)	
		R = horner_intrinsic(X, coeficientes, 10000);
	t2 = clock();
  	float diff_int = (((float)t2 - (float)t1) / CLOCKS_PER_SEC );
  	cout << "Time taken (Intrinsics): " << diff_int << endl;
		
	_mm_free(coeficientes);
	
	return 0;
}
