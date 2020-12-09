#include "vBF_GPU_kernel.h"

__global__ void naiveKernel(float* inputImage, float* outputImage, 
		 const uint32_t imageWidth,
		 const uint32_t imageHeight, const uint32_t numOfSpectral,
		 const uint32_t windowSize, const float sigmaInvR, 
		 const float sigmaInvD){

	int32_t idx = blockIdx.x*gridDim.x + threadIdx.x;
	int32_t idy = blockIdx.y*gridDim.y + threadIdx.y;
	rowStart = max(0, idx - windowSize);
	rowEnd = min(imageWidth, idx + windowSize);
	colStart = max(0, idy - windowSize);
	colEnd = min(imageHeight, idx + windowSize);
	float delta;
	for(uint32_t row = rowStart; row < rowEnd; ++ row){
		for(uint32_t col = colStart; col < colEnd; ++col){
			delta = 0;
			for(uint32_t k = 0; k < numOfSpectral; ++k){
				delta += pow(inputImage[k*imageWidth*imageHeight + row*imageWidth + col] - 
						inputImage[k*imageWidth*imageHeight + idx*imageHeight + idy], 2);
			}
			omega = exp(-delta*sigmaInvR- (pow(row - idx, 2) + pow(col - idy, 2))*sigmaInvD);
			y_d += omega;
			for(uint32_t k = 0; k < numOfSpectral; ++k){
				outputImage[k*imageWidth*imageHeight + idx*imageHeight + idy] += 
					inputImage[k*imageWidth*imageHeight + idx*imageHeight + idy]*omega;
			}
		}
	}
	for(uint32_t i = 0; i < numOfSpectral; ++i){
		outputImage[k*imageWidth*imageHeight + idx*imageHeight + idy] = outputImage[k*imageWidth*imageHeight + idx*imageHeight + idy]/y_d;	
	}

}

__global__ void vBF_kernel(float* inputImage, const int16_t imageWidth, const int16_t imageHeight, 
	const int16_t windowSize, const int16_t numOfSpectral,
	int16_t windowSize, const float sigmaInvR, const float sigmaInvD){

	extern __shared__ volatile float sharedMem1[];  
	__shared__ volatile float sharedMem2[gridDim.x];  
	float r[spectralPerThread];
	float y_n[spectralPerThread];
	float y_d[spectralPerThread];

	int8_t warpLane = threadIdx.x - ((threadIdx.x>>5)<<5); ;
	int8_t warpIdx = threadIdx.x>>5;

	//__shared__ int16_t sharedI_bar[warpSize];
	//__shared__ int16_t sharedJ_bar[warpSize];

	int16_t I_block = blockIdx.x*sizeA;
	int16_t J_block = blockIdx.y*sizeB;
	uint32_t I_warp = I_block + warpIdx/sizeB;
	uint32_t J_warp = J_block + warpIdx%sizeB;
	for(int16_t k = warpLane; k < numOfSpectral; k += warpSize){
		r[regId] = inputImage[(I_warp*imageWidth + J_warp)*warpSize + k];		
	}	
	int16_t I_blockMin = max(0, I_block - windowSize);
	int16_t J_blockMin = max(0, J_block - windowSize);
	uint8_t neighborWidth = min(I_block + heightA + windowSize, imageHeight) - max(0, I_block - windowSize);
	uint8_t neighborHeight = min(J_block + widthA + windowSize, imageWidth) - max(0, J_block - windowSize);
	uint16_t neighborSize = neighborWidth*neighborHeight;
	for(int16_t m = warpIdx; m < neighborSize; m+=warpSize){
		I_bar = max(0, I_block - windowSize) + m/(neighborWidth);	
		J_bar = max(0, J_block - windowSize) + m - (I_bar - I_blockMin)*neighborWidth;
		//sharedI_bar[warpIdx] = I_bar;
		//sharedJ_bar[warpIdx] = J_bar;
		if(m + warpIdx < neighborSize){
			for(int16_t k = warpLane; k < numOfSpectral; k+=warpSize){
				sharedMem1[warpIdx][k] = inputImage[(I_warp*imageWidth+J_warp)*numOfSpectral + k];
			}
		}
		__syncthreads();
		//in this kernel we define threadPerBlock = 1024, which means warpPerBlock == warpSize
		//This is the best way to make exponent function more efficient
		loopRange = min(neighborSize - m, warpPerBlock);	
		for(int16_t i = 0; i < loopRange; ++i){
			int16_t bufIdx = warpIdx*warpSize;
			//Reuse of I_bar and J_bar register
			//These register will be reset at line 50 
			I_bar = I_blockMin + (((m>>5) << 5) + i)/neighborWidth;
			J_bar = J_blockMin + (((m>>5) << 5) + i) - (I_bar - I_blockMin)*neighborWidth;
			if(I_warp - I_bar >= -windowSize && I_warp - I_bar <= windowSize && 
					J_warp - J_bar >= -windowSize && J_warp - J_bar <= windowSize){
				#pragma unroll
				for(int16_t k = warpLane; k < numOfSpectral; k += waprSize){
					delta += pow((sharedMem1[i*numOfSpectral+k] - r[k>>5]), 2);
				}
				__syncwarp();
				for (int offset = 16; offset > 0; offset /= 2)
					delta += __shfl_down_sync(FULL_MASK, delta, offset);
				if(warpLane == 0){
					//sharedMem2 is now delta buffer
					sharedMem2[bufIdx] = delta; 
				}
				__syncwarp();
			}
		}
		__syncthreads();
		if(sharedMem2[threadIdx.x] > 0){
			sharedMem2[threadIdx.x] = exp(-sharedMem2[threadIdx.x]*sigmaInvR
					- (pow((I_bar - I_warp), 2) + pow((J_bar - J_warp), 2))*sigmaInvD);
		}
		__syncwarp();
		for(int16_t i = 0; i < loopRange; ++i){
			if(sharedMem2[threadIdx.x] > 0){
				regId = 0;
				for(int16_t k = warpLane; k < numOfSpectral; k += warpSize){
					y_n[regId] += sharedMem2[(warpIdx<<5) + i]*sharedMem1[i*numOfSpectral + k];
				}
				yd += sharedMem2[(warpIdx<<5) + i];
			}
		}
		__syncthreads();
	}
	regId = 0;
	for(int16_t k = warpLane; k < numOfSpectral; k+=warpSize){
		outputImage[(I_warp*imageWidth + J_warp)*warpSize + k] = y_n[regId]/y_d;
		regId += 1;
	}
	
}

extern "C"

void vBF_GPU(float* inputImage, 
		float* outputImage,
		const int16_t imageWidth, 
		const int16_t imageHeight,  
		const int16_t windowSize, 
		const uint32_t numOfSpectral,
		int16_t windowSize,
	   	const float sigmaR, const float sigmaD){

	widthB = ceil((float imageWidth)/widthA); 
	heightB = ceil((float imageHeight)/heightA); 
	dim3 blockSize = dim3(widthB, heightB);	
	sigmaInvR = .5/pow(sigmaR, 2);
	sigmaInvD = .5/pow(sigmaD, 2);
	uint16_t sharedMem1Size = numOfSpectral*warpPerBlock;
	//thread num have to be 1024 
	//because we want warp per block equal to warp size
	vBF_kernel<<<blockSize, warpSize*warpSize, sharedMem1Size>>>(inputImage, width, numOfSpectral, 
			windowSize, sigmaInvR, sigmaInvD);	

}

void naiveGPU(float* inputImage, 
		float* outputImage,
		const int16_t imageWidth, 
		const int16_t imageHeight,  
		const int16_t windowSize, 
		const uint32_t numOfSpectral,
		int16_t windowSize,
	   	const float sigmaR, const float sigmaD){

	sigmaInvR = .5/pow(sigmaR, 2);
	sigmaInvD = .5/pow(sigmaD, 2);
}	

void CPUbenchmark(){

	//single thread
	for(int i = 0)

	//8 thread 
	#pragma parallel	


}
