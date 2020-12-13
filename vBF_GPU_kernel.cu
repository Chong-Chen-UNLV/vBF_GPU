#include "vBF_GPU.h"

__global__ void naiveKernel(float* inputImage, float* outputImage, 
		const int16_t windowSize, const float sigmaInvR, 
		const float sigmaInvD){

	int32_t colC = blockIdx.x*blockDim.x + threadIdx.x;
	int32_t rowC = blockIdx.y*blockDim.y + threadIdx.y;
	int32_t rowStart = max(0, rowC - windowSize);
	int32_t rowEnd = min(imageHeight - 1, rowC + windowSize);
	int32_t colStart = max(0, colC - windowSize);
	int32_t colEnd = min(imageWidth - 1, colC + windowSize);
	float delta, omega, y_d = 0;
	delta = 0;
	for(int32_t row = rowStart; row <= rowEnd; ++ row){
		for(int32_t col = colStart; col <= colEnd; ++col){
			delta = 0;
			for(int32_t k = 0; k < numOfSpectral; ++k){
				delta += pow(inputImage[k*imageWidth*imageHeight + row*imageWidth + col] - 
						inputImage[k*imageWidth*imageHeight + rowC*imageWidth + colC], 2);
			}
			omega = exp(-delta*sigmaInvR- (pow((float)row - rowC, 2) + pow((float)col - colC, 2))*sigmaInvD);
			y_d += omega;
			for(int32_t k = 0; k < numOfSpectral; ++k){
				outputImage[k*imageWidth*imageHeight + rowC*imageWidth + colC] += 
					inputImage[k*imageWidth*imageHeight + row*imageWidth + col]*omega;
			}
		}
	}
	for(int32_t k = 0; k < numOfSpectral; ++k){
		outputImage[k*imageWidth*imageHeight + rowC*imageWidth + colC] = 
			outputImage[k*imageWidth*imageHeight + rowC*imageWidth + colC]/y_d;	
	}

}

__global__ void vBF_kernel(float* inputImage, 
		float* outputImage,
		int16_t windowSize,
	   	const float sigmaInvR, const float sigmaInvD)
{

	extern __shared__ volatile float sharedMem1[];  
	__shared__ volatile float sharedMem2[warpPerBlock*warpSize];  
	
	float r[regSize] = {0};
	float y_n[regSize] = {0};
	float y_d = 0;

	int16_t warpLane = threadIdx.x - ((threadIdx.x>>5)<<5); ;
	int8_t warpIdx = threadIdx.x>>5;
	int16_t I_bar, J_bar;

	float delta = 0;
	//__shared__ int16_t sharedI_bar[warpSize];
	//__shared__ int16_t sharedJ_bar[warpSize];

	int16_t J_block = blockIdx.x*widthA;
	int16_t I_block = blockIdx.y*heightA;
	int32_t I_warp = I_block + warpIdx/widthA;
	int32_t J_warp = J_block + warpIdx%widthA;
	
	if(I_warp < imageWidth && J_warp < imageHeight){
		for(int16_t k = warpLane; k < numOfSpectral; k += warpSize){
			r[k>>5] = inputImage[(I_warp*imageWidth + J_warp)*numOfSpectral+ k];		
			__syncwarp();
		}	
		int16_t I_blockMin = max(0, I_block - windowSize);
		int16_t J_blockMin = max(0, J_block - windowSize);
		int16_t neighborWidth = min(J_block + heightA + windowSize + 1, imageWidth) - J_blockMin;
		int16_t neighborHeight = min(I_block + widthA + windowSize + 1, imageHeight) - I_blockMin;
		int16_t neighborSize = neighborWidth*neighborHeight;
		int16_t bufIdx; 
		uint8_t iterIdx = 0;

		for(int16_t m = warpIdx; m < neighborSize; m+=warpPerBlock){
			I_bar = I_blockMin + m/(neighborWidth);	
			J_bar = J_blockMin + m%neighborWidth;
			//sharedI_bar[warpIdx] = I_bar;
			//sharedJ_bar[warpIdx] = J_bar;
			if(m + warpIdx < neighborSize){
				for(int16_t k = warpLane; k < numOfSpectral; k+=warpSize){
					sharedMem1[warpIdx*numOfSpectral + k] = inputImage[(I_bar*imageWidth+J_bar)*numOfSpectral + k];
					__syncwarp();
				}
			}
			//in this kernel we define threadPerBlock = 1024, which means warpPerBlock == warpSize
			//This is the best way to make exponent function more efficient
			int16_t loopRange = min(neighborSize - m, warpPerBlock);	
			sharedMem2[threadIdx.x] = 0;
			bufIdx = warpIdx*warpSize;//multiply by warpSize
			iterIdx = m/warpPerBlock;
			__syncthreads();
			
			for(int16_t i = 0; i < loopRange; ++i){

				//Reuse of I_bar and J_bar register
				//These register will be reset at line 50 
				I_bar = I_blockMin + (iterIdx*warpPerBlock + i)/neighborWidth;
				J_bar = J_blockMin + (iterIdx*warpPerBlock + i)%neighborWidth;
				__syncwarp();
				if((I_warp - I_bar) >= -windowSize && (I_warp - I_bar) <= windowSize && 
						(J_warp - J_bar) >= -windowSize && (J_warp - J_bar) <= windowSize){
					delta = 0;
					__syncwarp();
#pragma unroll
					for(int16_t k = warpLane; k < numOfSpectral; k += warpSize){
						delta += pow((sharedMem1[i*numOfSpectral+k] - r[k>>5]), 2);
					}
					__syncwarp();
					for (int offset = 16; offset > 0; offset /= 2)
						delta += __shfl_down_sync(FULL_MASK, delta, offset);
					__syncwarp();
					if(warpLane == 0){
						//sharedMem2 is now delta buffer
						sharedMem2[bufIdx] = delta; 
					}
					__syncwarp();
				}
				bufIdx += 1;
				__syncwarp();
			}
			__syncthreads();
			if(sharedMem2[threadIdx.x] > 0){
				I_bar = I_blockMin + (iterIdx*warpPerBlock + warpLane)/neighborWidth;
				J_bar = J_blockMin + (iterIdx*warpPerBlock + warpLane)%neighborWidth;
				
				sharedMem2[threadIdx.x] = exp(-sharedMem2[threadIdx.x]*sigmaInvR
						- (pow(((float)I_bar - I_warp), 2) + pow(((float)J_bar - J_warp), 2))*sigmaInvD);
			}
			__syncwarp();
			for(int16_t i = 0; i < loopRange; ++i){
				if(sharedMem2[(warpIdx<<5) + i] > 0){
					for(int16_t k = warpLane; k < numOfSpectral; k += warpSize){
						y_n[k>>5] += sharedMem2[(warpIdx<<5) + i]*sharedMem1[i*numOfSpectral + k];
					}
					y_d += sharedMem2[(warpIdx<<5) + i];
				}
				__syncwarp();
			}
			__syncthreads();
		}
		//if(I_warp == 90 && J_warp == 60)
		//	I_warp = I_warp + warpIdx - threadIdx.x/warpSize ;
		for(int16_t k = warpLane; k < numOfSpectral; k+=warpSize){
			outputImage[(I_warp*imageWidth + J_warp)*numOfSpectral+ k] = (y_n[k>>5] + r[k>>5])/(y_d + 1);
		}
	}	
}




void vBF_GPU(float* inputImage, 
		float* outputImage,
		int16_t windowSize,
	   	const float sigmaR, const float sigmaD){
	
	uint32_t widthB = ceil(((float)imageWidth)/widthA); 
	uint32_t heightB = ceil(((float)imageHeight)/heightA); 
	dim3 blockSize = dim3(widthB, heightB, 1);	
	float sigmaInvR = .5/pow(sigmaR, 2);
	float sigmaInvD = .5/pow(sigmaD, 2);
	int sharedMem1Size = numOfSpectral*warpPerBlock;
	//thread num have to be 1024 
	//because we want warp per block equal to warp size
	
	vBF_kernel<<<blockSize, warpSize*warpPerBlock, sharedMem1Size*sizeof(float)>>>(inputImage, 
			outputImage, windowSize, sigmaInvR, sigmaInvD);	

}

void naiveGPU(float* inputImage, 
		float* outputImage,
		const int16_t windowSize, 
	   	const float sigmaR, const float sigmaD){

	float sigmaInvR = .5/pow(sigmaR, 2);
	float sigmaInvD = .5/pow(sigmaD, 2);
	int widthB = ceil(((float)imageWidth)/16);
	int heightB = ceil(((float)imageHeight)/16); 
	dim3 blockSize = dim3(widthB, heightB, 1);
	dim3 threadSize = dim3(16, 16, 1);
	naiveKernel<<<blockSize, threadSize>>>(inputImage, outputImage, 
			windowSize, sigmaInvR, sigmaInvD);
}	


