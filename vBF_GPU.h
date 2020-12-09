#ifndef VBFGPU_H 
#define VBFGPU_H

#define FULL_MASK 0xffffffff

void naiveGPU(float* inputImage, 
		float* outputImage,
		const int16_t imageWidth, 
		const int16_t imageHeight,  
		const int16_t windowSize, 
		const uint32_t numOfSpectral,
		int16_t windowSize,
	   	const float sigmaR, const float sigmaD);

void vBF_GPU(float* inputImage, 
		float* outputImage,
		const int16_t imageWidth, 
		const int16_t imageHeight,  
		const int16_t windowSize, 
		const uint32_t numOfSpectral,
		int16_t windowSize,
	   	const float sigmaR, const float sigmaD);


#endif
