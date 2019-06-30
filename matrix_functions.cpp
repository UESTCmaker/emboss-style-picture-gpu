#include "matrix_functions.h"
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//限定范围的随机均匀分布浮点数生成器
float random_Num(float maxNum, float minNum) {
	float randnum;
	int interval = maxNum - minNum + 1;
	do {
		randnum = rand() % interval + minNum;
	} while (randnum > maxNum || randnum < minNum);
	return randnum;
}

wbImage_t wbImage_new(int width, int height, int channels) {
	wbImage_t img = (wbImage_t)malloc(sizeof(wbImage));
	int i = 0;
	int total;
	float* p = NULL;
	img->height = height;
	img->width = width;
	img->channels = channels;
	img->pitch = ((width - 1) / GDDR5_DRAM_bursting + 1)*GDDR5_DRAM_bursting;
	total = height * width * channels;
	img->data = (float*)malloc(sizeof(float) * total);
	p = img->data;
	for (i; i < total; i++) {
		*(p + i) = random_Num(MAX, MIN);
	}
	return img;
}

wbImage_t wbImage_zero(int width, int height, int channels) {
	wbImage_t img = (wbImage_t)malloc(sizeof(wbImage));
	int i = 0;
	int total;
	float* p = NULL;
	img->height = height;
	img->width = width;
	img->channels = channels;
	img->pitch = ((width - 1) / GDDR5_DRAM_bursting + 1)*GDDR5_DRAM_bursting;
	total = height * width * channels;
	img->data = (float*)malloc(sizeof(float) * total);
	p = img->data;
	for (i; i < total; i++) {
		*(p + i) = 0.0;
	}
	return img;
}

void wbImage_display(wbImage_t img) {
	int i = 0, j = 0, index;
	for (i = 0; i < img->height; i++) {
		for (j = 0; j < img->width; j++) {
			index = i * (img->width) + j;
			printf("%d   ",int((img->data)[index]));
		}
		printf("\n");
	}
	printf("\n");
}


void wbImage_to_matrix(float* matrix,wbImage_t img) {
	int i = 0;
	int len = img->height *img->width *img->channels;
	float *pdata = img->data;
	for (i; i < len; i++) {
		*(matrix + i) = *(pdata + i);
	}
}

void matrix_to_wbImage(wbImage_t img, float* matrix) {
	int i = 0;
	int len = img->height *img->width *img->channels;
	float *pdata = img->data;
	for (i; i < len; i++) {
		 *(pdata + i) = *(matrix + i);
	}
}

void wbImage_delete(wbImage_t img) {
	free(img->data);
}

void matrix_display(float* img, unsigned int height, unsigned int width) {
	int i = 0, j = 0, index;
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			index = i * width + j;
			printf("%f\t", img[index]);
		}
		printf("\n");
	}
	printf("\n");
}