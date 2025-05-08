/* 
   CS485 GPU Cluster Programming
   Homework 4 MPI+CUDA on dot product
   10/21/2025
   Andrew Sohn
 */

 #include <iostream>
 using std::cerr;
 using std::endl;
 
 // System includes
 #include <stdio.h>
 #include <assert.h>
 
 // CUDA runtime
 #include <cuda_runtime.h>
 #include <cuda_profiler_api.h>
 
 // Helper functions and utilities to work with CUDA
 #include <unistd.h>
 #include <sys/time.h>
 
 #include "dot_decl.h"
 
 #define THREADS_PER_BLOCK 1024
 
 // Helper function to print an array of integers
 void print_array(const int *arr, int size, const char *label) {
     printf("%s: ", label);
     for (int i = 0; i < size; i++) {
         printf("%d ", arr[i]);
     }
     printf("\n");
 }
 
 int sum(int size, int *data) {
   int accum = 0;
   for (int i = 0; i < size; i++) {
     accum += data[i];
   }
   return accum;
 }
 
 int dev_my_log(int val) {
   int i, log_val = 0;
   for (i = val; i > 1; i = i >> 1)
     log_val++;
   return log_val;
 }
 
 int dot_product_cpu(int rank, int n, int *x, int *y) {
   int i = 0, j = 0;
   int prod = 0, bprod = 0;
   int lst_prods[1024];
 
   int nblks = n / THREADS_PER_BLOCK;
   int nthds = THREADS_PER_BLOCK;
   if (n < THREADS_PER_BLOCK) {
     nblks = 1;
     nthds = n;
   }
 
   for (i = 0; i < nblks; i++) {
     bprod = 0;
     for (j = 0; j < nthds; j++)
       bprod = bprod + *x++ * *y++;
     lst_prods[i] = bprod;
   }
 
   for (i = 0; i < nblks; i++)
     prod = prod + lst_prods[i];
 
   return prod;
 }
 
 // Accumulate using binary tree reduction
 __global__ void dot_prod_tree_reduction(int *a, int *b, int *c, int my_work, int log_n) {
     __shared__ int temp[THREADS_PER_BLOCK];
     int tid = threadIdx.x;
     int idx = blockIdx.x * blockDim.x + tid;
     if (idx < my_work) {
         temp[tid] = a[idx] * b[idx];
     } else {
         temp[tid] = 0;
     }
     __syncthreads();
     for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
          if (tid < stride) {
              temp[tid] += temp[tid + stride];
          }
          __syncthreads();
     }
     if (tid == 0) {
          c[blockIdx.x] = temp[0];
     }
 }
 
 // Thd 0 accumulates (serial version)
 __global__ void dot_prod_serial(int *a, int *b, int *c, int n, int my_work, int log_n) {
     int block_start = blockIdx.x * blockDim.x;
     int block_end = block_start + blockDim.x;
     if (block_end > my_work) {
         block_end = my_work;
     }
     int sumVal = 0;
     for (int i = block_start; i < block_end; i++) {
         sumVal += a[i] * b[i];
     }
     if (threadIdx.x == 0) {
         c[blockIdx.x] = sumVal;
     }
 }
 
 /**
  * Run simple dot product using CUDA
  */
 int dot_product_cuda(int my_rank, int my_work, int *h_A, int *h_B) {
   int i, cuda_prod = 0, indiv_cpu_prod = 0;
   int *d_A, *d_B, *d_C_nblks, *h_C_nblks;
   int blocks_per_grid = (my_work + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
   int threads_per_block = THREADS_PER_BLOCK;
   if (my_work < THREADS_PER_BLOCK) threads_per_block = my_work;
   int nblks = blocks_per_grid;
   int nthds = threads_per_block;
   h_C_nblks = (int *) malloc(sizeof(int) * nblks);
   for (i = 0; i < nblks; i++) h_C_nblks[i] = 0;
 
   cudaDeviceProp prop;
   cudaGetDeviceProperties(&prop, 0);
   if (my_rank == 0) {
       printf("\n**** properties: rank=%d *****\n", my_rank);
       printf("prop.name=%s\n", prop.name);
       printf("prop.multiProcessorCount=%d\n", prop.multiProcessorCount);
       printf("prop.major=%d minor=%d\n", prop.major, prop.minor);
       printf("prop.maxThreadsPerBlock=%d\n", prop.maxThreadsPerBlock);
       printf("maxThreadsDim.x=%d maxThreadsDim.y=%d maxThreadsDim.z=%d\n",
              prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
       printf("prop.maxGridSize.x=%d maxGridSize.y=%d maxGridSize.z=%d\n",
              prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
       printf("prop.maxThreadsPerMultiProcessor=%d\n", prop.maxThreadsPerMultiProcessor);
       printf("prop.totalGlobalMem=%u\n", prop.totalGlobalMem);
       printf("**** properties: rank=%d *****\n", my_rank);
       printf("\n");
   }
   unsigned int mem_size = sizeof(int) * my_work;
   printf("rank=%d: my_work=%d data_size=%d bytes\n", my_rank, my_work, mem_size);
 
   // Debug: print input arrays before copying to device
  //  printf("Debug: Host arrays before device copy:\n");
  //  print_array(h_A, my_work, "h_A");
  //  print_array(h_B, my_work, "h_B");
 
   cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size);
   cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size);
   cudaMalloc(reinterpret_cast<void **>(&d_C_nblks), sizeof(int) * nblks);
   cudaMemcpy(d_A, h_A, mem_size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, h_B, mem_size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_C_nblks, h_C_nblks, sizeof(int) * nblks, cudaMemcpyHostToDevice);
   int log_n = dev_my_log(nthds);
   printf("rank=%d: CUDA kernel launch with %d blocks of %d threads\n", my_rank, nblks, nthds);
   // Uncomment one of the kernel calls below as needed:
   dot_prod_tree_reduction<<<nblks, nthds>>>(d_A, d_B, d_C_nblks, my_work, log_n);
   dot_prod_serial<<<nblks, nthds>>>(d_A, d_B, d_C_nblks, nthds, my_work, log_n);
   cudaMemcpy(h_C_nblks, d_C_nblks, sizeof(int) * nblks, cudaMemcpyDeviceToHost);
 
   // Debug: print the partial product results from the device
  //  printf("Debug: Partial products array from device (h_C_nblks):\n");
  //  print_array(h_C_nblks, nblks, "h_C_nblks");
 
   fflush(stdout);
   cuda_prod = sum(nblks, h_C_nblks);
 
   // Note: dot_product_cpu will modify the pointers h_A and h_B, so if you need
   // to debug these arrays further, consider copying them before calling.
   indiv_cpu_prod = dot_product_cpu(my_rank, my_work, h_A, h_B);
   if (indiv_cpu_prod == cuda_prod)
     printf("TEST CPU %d: PASS: cuda_prod=%x == indiv_cpu_prod=%x\n", my_rank, cuda_prod, indiv_cpu_prod);
   else
     printf("TEST CPU %d: FAIL: cuda_prod=%x != indiv_cpu_prod=%x\n", my_rank, cuda_prod, indiv_cpu_prod);
   fflush(stdout);
   
   // Free device memory (optional cleanup)
   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C_nblks);
   free(h_C_nblks);
 
   return cuda_prod;
 }
 
 /* End of file */
 