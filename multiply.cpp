#include <stdio.h>
#include <time.h>
#include <cuda.h> 
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "matrix_functions.h"
#include "fileProcess_functions.h"

__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int Width, int Height);
__global__ void MatrixMulKernel_Tiles(float* Md, float* Nd, float* Pd, int Width, int Height);
wbImage_t MatrixMul_CPU(wbImage_t m1, wbImage_t m2);
wbImage_t MatrixMul_GPU(wbImage_t m1, wbImage_t m2);

wbImage_t MatrixMul_GPU(wbImage_t m1, wbImage_t m2) {
	wbImage_t m3 = wbImage_zero(M_HEIGHT, M_WIDTH, M_CHANNEL);
	printf("wbImage init success!\n");
	float *gpu_img1 = NULL;
	float *gpu_img2 = NULL;
	float *gpu_img3 = NULL;
	float* cpu_img1 = m1->data;
	float* cpu_img2 = m2->data;
	float* cpu_img3 = m3->data;
	dim3 threads_square(BLOCK_HEIGHT, BLOCK_WIDTH);
	dim3 blocks_square(GRID_HEIGHT, GRID_WIDTH);
	printf("BLOCK_HEIGHT：%d BLOCK_WIDTH：%d , GRID_HEIGHT：%d GRID_WIDTH：%d \n", BLOCK_HEIGHT, BLOCK_WIDTH, GRID_HEIGHT, GRID_WIDTH);
	//wbImage_display(m1);
	//wbImage_display(m2);
	unsigned int img1_size = m1->height * m1->width * m1->channels;
	unsigned int img2_size = m2->height * m2->width * m2->channels;
	unsigned int img3_size = m3->height * m3->width * m3->channels;

	cudaMalloc((void**)&gpu_img1, sizeof(float)*img1_size);
	cudaMalloc((void**)&gpu_img2, sizeof(float)*img2_size);
	cudaMalloc((void**)&gpu_img3, sizeof(float)*img3_size);

	//cudaMemcpy 将数据从cpu复制到device内存中
	cudaMemcpy(gpu_img1, cpu_img1, sizeof(float)* img1_size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_img2, cpu_img2, sizeof(float)* img2_size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_img3, cpu_img3, sizeof(float)* img3_size, cudaMemcpyHostToDevice);

	// 启动kernel函数
	MatrixMulKernel << <blocks_square, threads_square >> > (gpu_img1, gpu_img2, gpu_img3, M_WIDTH, M_HEIGHT);
	//MatrixMulKernel_GPU_Tiles <<<blocks_square, threads_square >>> (gpu_img1, gpu_img2, gpu_img3, M_WIDTH, M_HEIGHT);
	cudaMemcpy(cpu_img3, gpu_img3, sizeof(float)* img3_size, cudaMemcpyDeviceToHost);

	cudaFree(gpu_img1);
	cudaFree(gpu_img2);
	cudaFree(gpu_img3);

	//wbImage_display(m3);
	return m3;
}

__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int Width, int Height)
{
	// Calculate the row index of the Pd element and M
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	// Calculate the column index of Pd and N
	int Col = blockIdx.x*blockDim.x + threadIdx.x;
	float Pvalue = 0.0;
	// each thread computes one element of the block sub-matrix
	for (int k = 0; k < Height; ++k) {
		Pvalue += Md[Row*Width + k] * Nd[k*Width + Col];
	}
	Pd[Row*Width + Col] = Pvalue;
	//printf("thread[%d][%d]\t%f\t%f\t%f\n", Row, Col, Pvalue,Md[Row*Width+Col], Nd[Row*Width + Col]);
}


__global__ void MatrixMulKernel_Tiles(float* Md, float* Nd, float* Pd, int Width, int Height)
{
	__shared__ float Md_T[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nd_T[TILE_WIDTH][TILE_WIDTH];

	int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
	int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	float Pvalue = 0.0;
	// each thread computes one element of the block sub-matrix
	for (int k = 0; k < int((Width - 1) / TILE_WIDTH) + 1; k++) {
		if (k * TILE_WIDTH + tx < Width && Row < Height) {
			Md_T[ty][tx] = Md[Row*Width + k * TILE_WIDTH + tx];
		}
		else {
			Md_T[ty][tx] = 0.0;
		}
		if (k*TILE_WIDTH + ty < Height && Col < Width) {
			Nd_T[ty][tx] = Nd[(k*TILE_WIDTH + ty)* Width + Col];
		}
		else {
			Nd_T[ty][tx] = 0.0;
		}
		//__syncthreads();
		for (int i = 0; i < TILE_WIDTH; i++) {
			Pvalue += Md_T[ty][i] * Nd_T[i][tx];
			//if (Row == 0 && Col == 0) {
			//	printf("%d * %d + ", int(Md_T[ty][i]),int( Nd_T[i][tx]) );
			//}
		}
		//__syncthreads();
	}
	if (Row < Height && Col < Width) {
		Pd[Row*Width + Col] = Pvalue;
		//if (Row == 0 && Col == 0) {
		//	printf("= %d \n", int(Pvalue) );
		//}
	}
}


wbImage_t MatrixMul_CPU(wbImage_t m1, wbImage_t m2)
{
	wbImage_t m3 = wbImage_zero(M_HEIGHT, M_WIDTH, M_CHANNEL);
	printf("wbImage init success!\n");
	float* cpu_img1 = m1->data;
	float* cpu_img2 = m2->data;
	float* cpu_img3 = m3->data;
	int Row = 0;
	int Col = 0;
	int k = 0;
	float Pvalue = 0.0;
	for (Row = 0; Row < m1->height; Row++) {
		for (Col = 0; Col < m2->width; Col++) {
			Pvalue = 0.0;
			for (k = 0; k < m1->width; ++k) {
				Pvalue += *(cpu_img1 + Row * (m1->width) + k) * *(cpu_img2 + k * (m1->width) + Col);
			}
			*(cpu_img3 + Row * (m2->width) + Col) = Pvalue;
		}

	}
	return m3;
}