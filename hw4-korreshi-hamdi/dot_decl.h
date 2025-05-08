#ifndef DOT_DECL_H
#define DOT_DECL_H

#ifdef __cplusplus
extern "C" {
#endif

// Initializes an array with random values (0 to 15)
void init_vec(int *data, int dataSize);

// Computes the dot product on the host (CPU)
int dot_product_host(int nprocs, int n, int *x, int *y);

// Computes the base-2 logarithm of an integer
int my_log(int val);

// Computes the dot product on the GPU using CUDA
int dot_product_cuda(int my_rank, int my_work, int *h_A, int *h_B);

// Sums an integer array of given size
int sum(int size, int *data);

#ifdef __cplusplus
}
#endif

#endif // DOT_DECL_H
