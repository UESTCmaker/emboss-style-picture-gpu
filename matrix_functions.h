#pragma once
typedef struct {
	int width;
	int height;
	int pitch;
	int channels;
	float *data;
}wbImage, *wbImage_t;

#define GDDR5_DRAM_bursting 4
#define MAX 255.0
#define MIN 0.0

#define ksize 3


wbImage_t wbImage_new(int width, int height, int channels);
wbImage_t wbImage_zero( int width, int height, int channels);
//wbImage_t wbImport(char*File);

void wbImage_to_matrix(float* matrix, wbImage_t img);
void matrix_to_wbImage(wbImage_t img, float* matrix);
void wbImage_delete(wbImage_t img);
void wbImage_display(wbImage_t img);
void matrix_display(float* img, unsigned int height, unsigned int width);

#pragma once
