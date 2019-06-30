#include <stdio.h>
#include <time.h>
#include <cuda.h> 
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "matrix_functions.h"
#include "fileProcess_functions.h"

//测试用参数
#define M_HEIGHT 24
#define M_WIDTH 16
#define M_CHANNEL 1
#define M_SIZE M_HEIGHT*M_WIDTH*M_CHANNEL

//运行设置参数
#define BLOCK_HEIGHT 16
#define BLOCK_WIDTH  16
#define GRID_HEIGHT int(( M_HEIGHT - 1) / BLOCK_HEIGHT) + 1
#define GRID_WIDTH  int((M_WIDTH - 1) / BLOCK_WIDTH) + 1

#define TILE_WIDTH BLOCK_WIDTH
#define TILE_HEIGHT BLOCK_HEIGHT

void deviceInfo();
bool InitCUDA();

//浮雕运算卷积核
float core[ksize*ksize] = {
	-1.0,-1.0,0.0,
	-1.0,0.0,1.0,
	0.0,1.0,1.0
};

//模糊运算卷积核
//float core[ksize*ksize] = {
//		0.0,0.2,0.0,
//		0.2,0.2,0.2,
//		0.0,0.2,0.0
//	};

//使用常量内存存储卷积核
__constant__  float M[ksize*ksize];

//函数封装
__global__ void MatrixConvoKernel(float* P, float* N, int width, int height);
__global__ void MatrixConvoKernel_Tiles(float* P, float* N, int Width, int Height);
__global__ void MatrixConvoKernel_Tiles_Center(float* P, float* N, int Width, int Height);
wbImage_t MatrixConvo_GPU_Tiles_Center(wbImage_t m1);
wbImage_t MatrixConvo_GPU(wbImage_t img);
wbImage_t MatrixConvo_CPU(wbImage_t img);
wbImage_t MatrixConvo_GPU_Tiles(wbImage_t m1);

/*************************/
/*  测试用图像文件说明   */
/*   flower(6000*4000)   */
/*   concert(2880*1920)  */
/*   palace(1920*1080)   */
/*   moongray(1000*562)  */
/*************************/

char* file = ".\\img\\flower.jpg";
char* file_cpu = ".\\img\\palace_cpu.jpg";
char* file_gpu = ".\\img\\palace_gpu.jpg";
char* file_gpu_tiles = ".\\img\\palace_gpu_tiles.jpg";
char* file_gpu_tiles_center = ".\\img\\flower_gpu_tiles_center.jpg";

int main() {

	clock_t start,gpu_start, gpu_finish, gpu_tiles_start,gpu_tiles_finish, gpu_tiles_center_start, gpu_tiles_center_finish,cpu_start, cpu_finish, finish;	//定义第一次调用CPU时钟单位的实际，可以理解为定义一个计数器
	double Total_time,gpu_time,gpu_tiles_time,gpu_tiles_center_time,cpu_time;		//定义一个double类型的变量，用于存储时间单位
	start = clock();		//获取进入要测试执行时间代码段之前的CPU时间占用值
	printf("clock init success!\n");
	if (!InitCUDA()) { return 0; }//CUDA 初始化
	printf("CUDA init success!\n");

	wbImage_t img, output1,output2,output3,output4;

	//读取文件信息
	img = get_image_info(file);

	////执行GPU瓦片化并行图像处理程序
	//gpu_tiles_start = clock();
	//output1 = MatrixConvo_GPU_Tiles(img);
	//gpu_tiles_finish = clock();
	//gpu_tiles_time = (double)(gpu_tiles_finish - gpu_tiles_start) / CLOCKS_PER_SEC;
	//printf("GPU瓦片化程序执行时间：%.3fs\n", gpu_tiles_time);
	//save_image_info(output1, file_gpu_tiles);

	////执行CPU串行图像处理程序
	//cpu_start = clock();
	//output3 = MatrixConvo_CPU(img);
	//cpu_finish = clock();
	//cpu_time = (double)(cpu_finish - cpu_start) / CLOCKS_PER_SEC;
	//printf("CPU程序执行时间：%.3fs\n", cpu_time);
	//save_image_info(output3, file_cpu);

	////执行GPU并行图像处理程序
	//gpu_start = clock();
	//output2 = MatrixConvo_GPU(img);
	//gpu_finish = clock();
	//gpu_time = (double)(gpu_finish - gpu_start) / CLOCKS_PER_SEC;
	//printf("GPU程序执行时间：%.3fs\n", gpu_time);
	//save_image_info(output2, file_gpu);

	//执行GPU并行图像处理程序
	gpu_tiles_center_start = clock();
	output4 = MatrixConvo_GPU(img);
	gpu_tiles_center_finish = clock();
	gpu_tiles_center_time = (double)(gpu_tiles_center_finish - gpu_tiles_center_start) / CLOCKS_PER_SEC;
	printf("GPU瓦片化（仅加载中心元素）程序执行时间：%.3fs\n", gpu_tiles_center_time);
	//save_image_info(output4, file_gpu_tiles_center);

	finish = clock();
	Total_time = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("程序总执行时间：%.3fs\n", Total_time);

	wbImage_delete(img);
	//wbImage_delete(output1);
	//wbImage_delete(output2);
	//wbImage_delete(output3);
	return 0;
}

wbImage_t MatrixConvo_CPU(wbImage_t img) {
	wbImage_t output = (wbImage_t)malloc(sizeof(wbImage));
	int height = img->height;
	int width = img->width;
	int i, j, k, x, y;
	float *outp = NULL, *inp = img->data;
	output->height = height;
	output->width = width;
	output->channels = img->channels;
	output->data = (float*)malloc(sizeof(float)*height*width);
	outp = output->data;
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			*(outp + i * width + j) = 0;
			for (k = 0; k < ksize*ksize; k++) {
				//推导卷积核要累乘加运算的原始矩阵各元素坐标，以i,j为卷积核累乘加运算对应的原始矩阵的中心点
				y = i + k / ksize - ksize / 2 ;
				x = j + k % ksize - ksize / 2 ;
				if (y < 0 || y >= height || x < 0 || x >= width) {
					continue;
					//outp[i*width + j] += core[k] * 0;
				}
				else {
					*(outp + i * width + j) += core[k] * (*(inp + y * width + x));
				}
			}
			//printf("%f\n", *(outp + i * width + j));
		}
	}
	return output;
}

wbImage_t MatrixConvo_GPU_Tiles(wbImage_t m1) {
	wbImage_t m2 = wbImage_zero(m1->width, m1->height, m1->channels);
	printf("wbImage init success!\n");
	float *gpu_img1 = NULL;
	float *gpu_img2 = NULL;
	float* cpu_img1 = m1->data;
	float* cpu_img2 = m2->data;
	dim3 threads_square(BLOCK_WIDTH, BLOCK_HEIGHT);
	dim3 blocks_square(int((m1->width - 1) / BLOCK_WIDTH) + 1, int((m1->height - 1) / BLOCK_HEIGHT) + 1);
	printf("BLOCK_HEIGHT：%d BLOCK_WIDTH：%d , GRID_HEIGHT：%d GRID_WIDTH：%d \n", BLOCK_HEIGHT, BLOCK_WIDTH, int((m1->height - 1) / BLOCK_HEIGHT) + 1, int((m1->width - 1) / BLOCK_WIDTH) + 1);

	unsigned int img1_size = m1->height * m1->width * m1->channels;
	unsigned int img2_size = m2->height * m2->width * m2->channels;

	cudaMalloc((void**)&gpu_img1, sizeof(float)*img1_size);
	cudaMalloc((void**)&gpu_img2, sizeof(float)*img2_size);

	//cudaMemcpy 将数据从cpu复制到device内存中
	cudaMemcpy(gpu_img1, cpu_img1, sizeof(float)* img1_size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_img2, cpu_img2, sizeof(float)* img2_size, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(M, core, ksize*ksize * sizeof(float));
	// 启动kernel函数
	printf("width:%d height:%d\n", m2->width, m2->height);
	MatrixConvoKernel_Tiles << <blocks_square, threads_square >> > (gpu_img1, gpu_img2, m1->width, m1->height);
	cudaMemcpy(cpu_img2, gpu_img2, sizeof(float)* img2_size, cudaMemcpyDeviceToHost);

	cudaFree(gpu_img1);
	cudaFree(gpu_img2);

	return m2;

}

__global__ void MatrixConvoKernel_Tiles(float* P, float* N, int Width, int Height) {
	__shared__ float partial[TILE_HEIGHT + ksize - 1][TILE_WIDTH + ksize - 1];
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int halo_index_left = (blockIdx.x - 1)*blockDim.x + threadIdx.x;
	int halo_index_right = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
	int halo_index_up = (blockIdx.y - 1)*blockDim.y + threadIdx.y;
	int halo_index_down = (blockIdx.y + 1)*blockDim.y + threadIdx.y;
	int n = ksize / 2;
	int base_x = threadIdx.x + n;
	int base_y = threadIdx.y + n;
	int k = 0, offset_x = 0, offset_y = 0;
	float Pvalue = 0.0;

	if (threadIdx.x >= blockDim.x - n) {
		//加载左侧上方部分光环元素
		if (threadIdx.y >= blockDim.y - n) {
			partial[base_y - blockDim.y][base_x - blockDim.x] = (halo_index_left >= 0) && (halo_index_up >= 0) ? P[halo_index_up*Width + halo_index_left] : 0.0;
		}

		//加载左侧中间部分光环元素
		partial[base_y][base_x - blockDim.x] = (halo_index_left >= 0) && (Row < Height) ? P[Row*Width + halo_index_left] : 0.0;

		//加载左侧下方部分光环元素
		if (threadIdx.y < n) {
			partial[base_y + blockDim.y][base_x - blockDim.x] = (halo_index_left >= 0) && (halo_index_down < Height) ? P[halo_index_down*Width + halo_index_left] : 0.0;
		}
	}
	else if (threadIdx.x < n) {

		//加载右侧上方部分光环元素
		if (threadIdx.y >= blockDim.y - n) {
			partial[base_y - blockDim.y][base_x + blockDim.x] = (halo_index_right < Width) && (halo_index_up >= 0) ? P[halo_index_up*Width + halo_index_right] : 0.0;
		}

		//加载右侧中间部分光环元素
		partial[base_y][base_x + blockDim.x] = (halo_index_right < Width) && (Row < Height) ? P[Row*Width + halo_index_right] : 0.0;

		//加载右侧下方部分光环元素
		if (threadIdx.y < n) {
			partial[base_y + blockDim.y][base_x + blockDim.x] = (halo_index_right < Width) && (halo_index_down < Height) ? P[halo_index_down*Width + halo_index_right] : 0.0;
		}
	}

	//加载上侧中间部分光环元素
	if (threadIdx.y >= blockDim.y - n) {
		partial[base_y - blockDim.y][base_x] = (halo_index_up >= 0) && (Col < Width) ? P[halo_index_up*Width + Col] : 0.0;
	}
	//加载下侧中间部分光环元素
	else if (threadIdx.y < n) {
		partial[base_y + blockDim.y][base_x] = (halo_index_down < Height) && (Col < Width) ? P[halo_index_down*Width + Col] : 0.0;
	}

	//加载中心部分元素
	if (Row < Height && Col < Width) {
		partial[base_y][base_x] = P[Row*Width + Col];
	}
	else {
		partial[base_y][base_x] = 0.0;
	}

	//同步瓦片数据
	__syncthreads();

	if (Row < Height && Col < Width) {
		for (k = 0; k < ksize*ksize; k++) {
			offset_x = base_x + k % ksize - ksize / 2;
			offset_y = base_y + k / ksize - ksize / 2;
			Pvalue += M[k] * partial[offset_y][offset_x];
		}
		N[Row*Width + Col] = Pvalue;
	}
}

wbImage_t MatrixConvo_GPU_Tiles_Center(wbImage_t m1) {
	wbImage_t m2 = wbImage_zero(m1->width, m1->height, m1->channels);
	printf("wbImage init success!\n");
	float *gpu_img1 = NULL;
	float *gpu_img2 = NULL;
	float* cpu_img1 = m1->data;
	float* cpu_img2 = m2->data;
	dim3 threads_square(BLOCK_WIDTH, BLOCK_HEIGHT);
	dim3 blocks_square(int((m1->width - 1) / BLOCK_WIDTH) + 1, int((m1->height - 1) / BLOCK_HEIGHT) + 1);
	printf("BLOCK_HEIGHT：%d BLOCK_WIDTH：%d , GRID_HEIGHT：%d GRID_WIDTH：%d \n", BLOCK_HEIGHT, BLOCK_WIDTH, int((m1->height - 1) / BLOCK_HEIGHT) + 1, int((m1->width - 1) / BLOCK_WIDTH) + 1);

	unsigned int img1_size = m1->height * m1->width * m1->channels;
	unsigned int img2_size = m2->height * m2->width * m2->channels;

	cudaMalloc((void**)&gpu_img1, sizeof(float)*img1_size);
	cudaMalloc((void**)&gpu_img2, sizeof(float)*img2_size);

	//cudaMemcpy 将数据从cpu复制到device内存中
	cudaMemcpy(gpu_img1, cpu_img1, sizeof(float)* img1_size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_img2, cpu_img2, sizeof(float)* img2_size, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(M, core, ksize*ksize * sizeof(float));
	// 启动kernel函数
	printf("width:%d height:%d\n", m2->width, m2->height);
	MatrixConvoKernel_Tiles_Center << <blocks_square, threads_square >> > (gpu_img1, gpu_img2, m1->width, m1->height);
	cudaMemcpy(cpu_img2, gpu_img2, sizeof(float)* img2_size, cudaMemcpyDeviceToHost);

	cudaFree(gpu_img1);
	cudaFree(gpu_img2);

	return m2;

}

__global__ void MatrixConvoKernel_Tiles_Center(float* P, float* N, int Width, int Height) {
	__shared__ float partial[TILE_HEIGHT][TILE_WIDTH];
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int base_x = threadIdx.x;
	int base_y = threadIdx.y;
	int k = 0, offset_x = 0, offset_y = 0,global_x=0,global_y=0;
	float Pvalue = 0.0;

	//加载中心部分元素
	if (Row < Height && Col < Width) {
		partial[base_y][base_x] = P[Row*Width + Col];
	}
	else {
		partial[base_y][base_x] = 0.0;
	}

	//同步瓦片数据
	__syncthreads();

	if (Row < Height && Col < Width) {
		for (k = 0; k < ksize*ksize; k++) {
			offset_x = base_x + k % ksize - ksize / 2;
			offset_y = base_y + k / ksize - ksize / 2;
			if (offset_x < 0 || offset_x >= TILE_WIDTH || offset_y < 0 || offset_y >= TILE_HEIGHT) {
				global_x = blockIdx.x * blockDim.x + offset_x;
				global_y = blockIdx.y * blockDim.y + offset_y;
				if (global_x < 0 || global_x >= Width || global_y < 0 || global_y >= Height) {
					continue;
				}
				else {
					Pvalue += M[k] * P[global_y*Width + global_x];
				}
			}
			else {
				Pvalue += M[k] * partial[offset_y][offset_x];
			}
		}
		N[Row*Width + Col] = Pvalue;
	}
}

wbImage_t MatrixConvo_GPU(wbImage_t m1){
	wbImage_t m2 = wbImage_zero( m1->width, m1->height, m1->channels);
	printf("wbImage init success!\n");
	float *gpu_img1 = NULL;
	float *gpu_img2 = NULL;
	float* cpu_img1 = m1->data;
	float* cpu_img2 = m2->data;
	dim3 threads_square(BLOCK_WIDTH, BLOCK_HEIGHT);
	dim3 blocks_square(int((m1->width - 1) / BLOCK_WIDTH) + 1, int((m1->height - 1) / BLOCK_HEIGHT) + 1);
	printf("BLOCK_HEIGHT：%d BLOCK_WIDTH：%d , GRID_HEIGHT：%d GRID_WIDTH：%d \n", BLOCK_HEIGHT, BLOCK_WIDTH, int((m1->height - 1) / BLOCK_HEIGHT) + 1, int((m1->width - 1) / BLOCK_WIDTH) + 1);

	unsigned int img1_size = m1->height * m1->width * m1->channels;
	unsigned int img2_size = m2->height * m2->width * m2->channels;

	cudaMalloc((void**)&gpu_img1, sizeof(float)*img1_size);
	cudaMalloc((void**)&gpu_img2, sizeof(float)*img2_size);

	//cudaMemcpy 将数据从cpu复制到device内存中
	cudaMemcpy(gpu_img1, cpu_img1, sizeof(float)* img1_size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_img2, cpu_img2, sizeof(float)* img2_size, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(M,core,ksize*ksize*sizeof(float));
	// 启动kernel函数
	printf("width:%d height:%d\n", m2->width, m2->height);
	MatrixConvoKernel<< <blocks_square, threads_square >> > (gpu_img1,gpu_img2, m2->width, m2->height);
	cudaMemcpy(cpu_img2, gpu_img2, sizeof(float)* img2_size, cudaMemcpyDeviceToHost);

	cudaFree(gpu_img1);
	cudaFree(gpu_img2);

	return m2;

}

__global__ void MatrixConvoKernel(float* P, float* N, int Width, int Height) {
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int k = 0, offset_x = 0, offset_y = 0;
	float Pvalue = 0.0;
	//printf("Row:%d,Col:%d,Height:%d\n", Row, Col,Height);
	if (Row < Height && Col < Width) {
		//printf("Row:%d,Col:%d\n", Row, Col);
		for (k = 0; k < ksize*ksize; k++) {
			offset_y = Row + k / ksize - ksize / 2;
			offset_x = Col + k % ksize - ksize / 2;
			if (offset_x < 0 || offset_x >= Width || offset_y < 0 || offset_y >= Height) {
				continue;
			}
			else {
				Pvalue += M[k] * P[offset_y*Width + offset_x];
			}

		}
		N[Row*Width + Col] = Pvalue;
		//printf("thread[%d][%d]\t%f\n", Row, Col, N[Row*Width + Col]);
	}
}


//CUDA 初始化
bool InitCUDA() {
	int count;
	cudaGetDeviceCount(&count);//取得支持Cuda的装置的数目
	if (count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}
	int i;
	for (i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) {
				break;
			}
		}
	}
	if (i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}
	cudaSetDevice(i);
	return true;
}

void deviceInfo()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	int dev;
	for (dev = 0; dev < deviceCount; dev++)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		if (dev == 0)
		{
			if (/*deviceProp.major==9999 && */deviceProp.minor = 9999 && deviceProp.major == 9999)
				printf("\n");

		}
		printf("\nDevice%d:\"%s\"\n", dev, deviceProp.name);
		printf("Total amount of global memory                   %u bytes\n", deviceProp.totalGlobalMem);
		printf("Number of mltiprocessors                        %d\n", deviceProp.multiProcessorCount);
		printf("Total amount of constant memory:                %u bytes\n", deviceProp.totalConstMem);
		printf("Total amount of shared memory per block         %u bytes\n", deviceProp.sharedMemPerBlock);
		printf("Total number of registers available per block:  %d\n", deviceProp.regsPerBlock);
		printf("Warp size                                       %d\n", deviceProp.warpSize);
		printf("Maximum number of threada per block:            %d\n", deviceProp.maxThreadsPerBlock);
		printf("Maximum sizes of each dimension of a block:     %d x %d x %d\n", deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("Maximum size of each dimension of a grid:       %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		printf("Maximum memory pitch :                          %u bytes\n", deviceProp.memPitch);
		printf("Texture alignmemt                               %u bytes\n", deviceProp.texturePitchAlignment);
		printf("Clock rate                                      %.2f GHz\n", deviceProp.clockRate*1e-6f);
	}
}