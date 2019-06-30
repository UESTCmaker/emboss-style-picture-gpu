#include "fileProcess_functions.h"
#include "matrix_functions.h"
#include <stdio.h>  
#include <stdlib.h>
#include <math.h>
#include <atlimage.h>

wbImage_t get_image_info(char* imgFilePath) {
	CImage rawdata;
	wbImage_t img = (wbImage_t)malloc(sizeof(wbImage));
	if (!PathFileExists(imgFilePath)) {
		perror("Unrecognized file path:");
		return NULL;
	}
	printf("Check success!\n");
	rawdata.Load(imgFilePath);
	printf("Load success!\n");
	img->height =  rawdata.GetHeight();
	img->width = rawdata.GetWidth();
	img->channels = 1;

	unsigned int total = img->height * img->width * img->channels;
	img->data = (float*)malloc(sizeof(float) * total);
	printf("%d malloc success!\n",total);
	float* p = img->data;
	int i, j;

	for (i = 0; i < img->height; i++) {
		for (j = 0; j < img->width; j++) {
			COLORREF pixel = rawdata.GetPixel(j, i);
			*(p+i*(img->width) + j) = (int)GetRValue(pixel);
		}
	}
	return img;
}

void save_image_info(wbImage_t img,char* imgFilePath) {
	CImage prodata;
	prodata.Create(img->width, img->height, 24, 0);
	float* p = img->data;
	int i, j;
	printf("loop\n");
	for (i = 0; i < img->height; i++) {
		for (j = 0; j < img->width; j++) {
			//计算浮雕效果时需要给pixel加上128灰度偏移
			int pixel = *(p + i * (img->width) + j)+128; 
			if (pixel > 255) {
				pixel = 255;
			}
			else if (pixel < 0) {
				pixel = 0;
			}
			COLORREF color = RGB(pixel, pixel, pixel);
			prodata.SetPixel(j,i,color);
		}
	}
	prodata.Save(imgFilePath);
	printf("loop finish\n");
}