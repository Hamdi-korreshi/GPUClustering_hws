/*
  MPI+CUDA
  CS485 GPU Cluster Programming
  HW6 Convolution
  11/1/2025
  Andrew Sohn
  modified by Hamdi Korreshi
 */

 #include <iostream>
 using std::cerr;
 using std::endl;
 
 #include <stdio.h>
 #include <assert.h>
 #include <cuda_runtime.h>
 #include <cuda_profiler_api.h>
 
 #include <unistd.h>
 #include <sys/time.h>
 
 extern "C" {
   int conv_dev(int nprocs, int my_rank, int my_work, int *in_image,int *out_image, int height, int width, int filter_dim,int *filter_cpu);
 }
 
 #define THREADS_PER_BLOCK 256
 #define TILE_WIDTH 4
 #define MAX_TILE_WIDTH 16
 #define MAX_TILE MAX_TILE_WIDTH
 #define FILTER_DIM 3
 #define FILTER_RADIUS 1
 #define MAX_MASK_DIM 5
 
 // constant memory for filter
 __constant__ int filter_dev[FILTER_DIM*FILTER_DIM];
 
 void conv_host_cpu(int my_rank, int my_work, int* input, int* output, unsigned int height, unsigned int width, int *filter_cpu) {
   int offset = my_rank*my_work;
   int cnt=0;
   int R = FILTER_DIM/2;
   for(int out_row=offset; out_row<offset+my_work; out_row++) {
     for(int out_col=0; out_col<width; out_col++) {
       int sum = 0;
       // --- begin host slice convolution ---
       for(int fr=-R; fr<=R; fr++){
         for(int fc=-R; fc<=R; fc++){
           int ir = out_row + fr, ic = out_col + fc;
           if(ir>=0 && ir<height && ic>=0 && ic<width){
             sum += input[ir*width + ic]
                  * filter_cpu[(fr+R)*FILTER_DIM + (fc+R)];
           }
         }
       }
       // --- end host slice convolution ---
       output[cnt++] = sum;
     }
   }
 }
 
 int compare_cpu(int my_rank, int my_work, int *host, int *dev, int height, int width) {
   for (int i=0; i<my_work*width; i++) {
     if (dev[i] != host[i]) {
       printf("DIFFERENT: rank=%d: dev[%d]=%d != host[%d]=%d\n",
              my_rank,i,dev[i],i,host[i]);
       return 0;
     }
   }
   return 1;
 }
 
 __global__ void conv_dev_cuda(int my_rank,int my_work, int height,int width, int *input, int *output){
   int R = FILTER_DIM/2;
   int out_row = blockIdx.y * blockDim.y + threadIdx.y + my_rank*my_work;
   int out_col = blockIdx.x * blockDim.x + threadIdx.x;
   if(out_row < height && out_col < width) {
     int sum = 0;
     // --- begin GPU convolution ---
     for(int fr=-R; fr<=R; fr++){
       for(int fc=-R; fc<=R; fc++){
         int ir = out_row + fr, ic = out_col + fc;
         if(ir>=0 && ir<height && ic>=0 && ic<width){
           int f = filter_dev[(fr+R)*FILTER_DIM + (fc+R)];
           sum += input[ir*width + ic] * f;
         }
       }
     }
     // --- end GPU convolution ---
     output[out_row*width + out_col] = sum;
   }
 }
 
 void print_lst_cpu(int name,int rank,int n, int *l){
   int i=0;
   printf("CPU rank=%d: %d: size=%d:: ",rank,name,n);
   for (i=0; i<n; i++) printf("%x ",l[i]);
   printf("\n");
 }
 
 void print_filter_cpu(int name,int rank,int n, int *buf){
   int i=0,j;
   printf("CPU rank=%d: %d: size=%d:: ",rank,name,n);
   for (i=0; i<n; i++) 
     for (j=0; j<n; j++) printf("%x ",*buf++); /* buf[i][j]); */
   printf("\n");
 }
 
 void init_filter(int *buf) {
   int i,j,cnt=0;
   for (i=0; i<FILTER_DIM; i++) 
     for (j=0; j<FILTER_DIM; j++) *buf++ = cnt++;
 }
 
 int conv_dev(int nprocs, int my_rank, int my_work, int *h_in_image,int *h_out_image, int height, int width, int filter_dim,int *filter_cpu) {
   int in_size=height*width, out_size=in_size;
   int filter_size_bytes = filter_dim*filter_dim*sizeof(int);
 
   // --- copy filter to const memory ---
   cudaMemcpyToSymbol(filter_dev, filter_cpu, filter_size_bytes);
 
   int *d_in_image, *d_out_image;
   struct timeval timecheck;
 
   unsigned int in_bytes = sizeof(int) * in_size;
   unsigned int out_bytes = sizeof(int) * out_size;
   unsigned int my_work_bytes = sizeof(int) * my_work * width;
 
   // allocate
   cudaMalloc(&d_in_image,  in_bytes);
   cudaMalloc(&d_out_image, out_bytes);
 
   // upload input + zero output
   cudaMemcpy(d_in_image, h_in_image, in_bytes, cudaMemcpyHostToDevice);
   cudaMemset(d_out_image, 0, out_bytes);
 
   int bx_dim = TILE_WIDTH, by_dim = TILE_WIDTH;
   int gx_dim = (width  + bx_dim -1)/bx_dim;
   int gy_dim = (my_work + by_dim -1)/by_dim;
   dim3 grid(gx_dim,gy_dim), threads(bx_dim,by_dim);
   printf("rank=%d: CUDA kernel launch: grid(%d,%d), block(%d,%d)\n",
          my_rank, gx_dim, gy_dim, bx_dim, by_dim);
 
   // --- launch GPU kernel ---
   conv_dev_cuda<<<grid,threads>>>(my_rank,my_work,height,width,d_in_image,d_out_image);
   cudaDeviceSynchronize();
 
   // --- copy back this rank’s slice ---
   int offset = my_rank * my_work * width;
   int *h_slice = (int*)malloc(my_work_bytes);
   cudaMemcpy(h_slice, d_out_image + offset, my_work_bytes, cudaMemcpyDeviceToHost);
   memcpy(h_out_image + offset, h_slice, my_work_bytes);
 
   // --- compute and compare on CPU slice ---
   int *h_ref = (int*)malloc(my_work_bytes);
   conv_host_cpu(my_rank,my_work,h_in_image,h_ref,height,width,filter_cpu);
   printf("CPU %d: %s\n",
          my_rank,
          compare_cpu(my_rank,my_work,h_ref,h_slice,height,width) ? "PASS" : "FAIL");
 
   free(h_slice);
   free(h_ref);
   cudaFree(d_in_image);
   cudaFree(d_out_image);
 
   return 0;
 }
 