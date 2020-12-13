#ifndef VBFGPU_H 
#define VBFGPU_H
#include <stdint.h>

#define FULL_MASK 0xffffffff

const uint32_t widthA = 4;
const uint32_t heightA = 4;
const int warpSize = 32;
const uint16_t warpPerBlock = widthA*heightA;
const uint16_t imageWidth = 307;
const uint16_t imageHeight = 1280;
const uint16_t numOfSpectral = 96;
const uint16_t regSize = 3;

void naiveGPU(float* inputImage, 
		float* outputImage,
		const int16_t windowSize, 
	   	const float sigmaR, const float sigmaD);

void vBF_GPU(float* inputImage, 
		float* outputImage,
		const int16_t windowSize,
	   	const float sigmaR, const float sigmaD);
#endif
