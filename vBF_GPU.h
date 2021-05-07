#ifndef VBFGPU_H 
#define VBFGPU_H
#include <stdint.h>

#define FULL_MASK 0xffffffff

#define widthA 4
#define heightA 4
#define warpSize  32
#define warpPerBlock  widthA*heightA
const int imageWidth  = 307; 
const int imageHeight =  1280;
const int numOfSpectral = 191; 
#define regSize 6 

void naiveGPU(float* inputImage, 
		float* outputImage,
		const int16_t windowSize, 
	   	const float sigmaR, const float sigmaD);

void vBF_GPU(float* inputImage, 
		float* outputImage,
		const int16_t windowSize,
	   	const float sigmaR, const float sigmaD);
#endif
