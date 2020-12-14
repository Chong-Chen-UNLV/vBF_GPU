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
	__shared__ volatile float sharedMem3[warpPerBlock*warpSize];  
	
	float r[regSize] = {0};
	float y_n[regSize] = {0};
	float y_d = 0;

	int16_t warpLane = threadIdx.x - ((threadIdx.x>>5)<<5); ;
	int8_t warpIdx = threadIdx.x>>5;

	float delta = 0;
	__shared__ int16_t sharedI_bar[warpSize];
	__shared__ int16_t sharedJ_bar[warpSize];
	__shared__ int16_t J_block;
	__shared__ int16_t I_block;
	int16_t I_warp; 
	int16_t J_warp;
	uint16_t ii;
	if(threadIdx.x == 0){
		J_block = blockIdx.x*widthA;
		I_block = blockIdx.y*heightA;
	}
	__syncthreads();
	I_warp = I_block + warpIdx/widthA;
	J_warp = J_block + warpIdx%widthA;
	__syncwarp();
	
	if(I_warp < imageHeight && J_warp < imageWidth){
		for(int16_t k = warpLane; k < numOfSpectral; k += warpSize){
			r[k>>5] = inputImage[(I_warp*imageWidth + J_warp)*numOfSpectral+ k];		
			__syncwarp();
		}	
		int16_t I_blockMin = I_block - windowSize < 0 ? 0 : I_block-windowSize;
		int16_t J_blockMin =  J_block - windowSize < 0 ? 0 : J_block-windowSize;
		int16_t neighborWidth = ((J_block + widthA + windowSize) > imageWidth ? imageWidth :
				(J_block + widthA + windowSize)) - J_blockMin;
			
		int16_t neighborHeight = ((I_block + heightA + windowSize) > imageHeight ? imageHeight:
				(I_block + heightA + windowSize)) - I_blockMin;
		int16_t neighborSize = neighborWidth*neighborHeight;
		int16_t bufIdx; 

		for(int16_t m = warpIdx; m < neighborSize; m+=warpPerBlock){
			if(warpLane == 0){
				sharedI_bar[warpIdx] = I_blockMin + m/(neighborWidth);
				sharedJ_bar[warpIdx] = J_blockMin + m%neighborWidth;
			}
			__syncthreads();
			if(warpLane < warpPerBlock && m/warpPerBlock + warpLane < neighborSize){
				if(abs(sharedI_bar[warpLane] - I_warp) <= windowSize &&
						abs(sharedJ_bar[warpLane] - J_warp) <= windowSize){
					sharedMem3[threadIdx.x] = 
						(sharedI_bar[warpLane] - I_warp)*(sharedI_bar[warpLane] - I_warp)
						+ (sharedJ_bar[warpLane] - J_warp)*(sharedJ_bar[warpLane] - J_warp);	
				}
			}
			sharedMem1[warpIdx*numOfSpectral + warpLane] = 
				inputImage[(sharedI_bar[warpIdx]*imageWidth+sharedJ_bar[warpIdx])*numOfSpectral + warpLane];
			sharedMem1[warpIdx*numOfSpectral + warpSize + warpLane] = 
				inputImage[(sharedI_bar[warpIdx]*imageWidth+sharedJ_bar[warpIdx])*numOfSpectral + warpSize +  warpLane];
			sharedMem1[warpIdx*numOfSpectral + 2*warpSize + warpLane] = 
				inputImage[(sharedI_bar[warpIdx]*imageWidth+sharedJ_bar[warpIdx])*numOfSpectral + 2*warpSize +  warpLane];
			//for(int16_t k = warpLane; k < numOfSpectral; k+=warpSize){
			//	sharedMem1[warpIdx*numOfSpectral + k] = inputImage[(sharedI_bar[warpIdx]*imageWidth+sharedJ_bar[warpIdx])*numOfSpectral + k];
			//}
			//in this kernel we define threadPerBlock = 1024, which means warpPerBlock == warpSize
			//This is the best way to make exponent function more efficient
			bufIdx = warpIdx*warpSize;//multiply by warpSize
			__syncthreads();
			#pragma unroll	
			for(int16_t i = 0; i < 16; ++i){
				if(sharedMem3[(warpIdx<<5)+ i]>0){
					ii = i*numOfSpectral+warpLane;
					delta = 0;
					delta += (sharedMem1[ii] - r[0])*
						(sharedMem1[ii] - r[0]);
					ii+=warpSize;
					delta += (sharedMem1[ii] - r[1])*
						(sharedMem1[ii] - r[1]);
					ii+=warpSize;
					delta += (sharedMem1[ii] - r[2])*
						(sharedMem1[ii] - r[2]);
					__syncwarp();
					for (uint8_t offset = 16; offset > 0; (offset=(offset>>1)))
						delta += __shfl_down_sync(FULL_MASK, delta, offset);
					if(warpLane == 0){
						//sharedMem2 is now delta buffer
						sharedMem2[bufIdx] = delta; 
					}
				}
				bufIdx += 1;
				__syncwarp();
			}
			//__syncthreads();
			if(sharedMem2[threadIdx.x] > 0){
				sharedMem2[threadIdx.x] = expf(-sharedMem2[threadIdx.x]*sigmaInvR
						- sharedMem3[threadIdx.x]*sigmaInvD);
			}
			__syncwarp();
			#pragma unroll
			for(int16_t i = 0; i < warpPerBlock; ++i){
				if(sharedMem2[(warpIdx<<5) + i] > 0){
					ii = i*numOfSpectral+warpLane;
					y_n[0] += sharedMem2[(warpIdx<<5) + i]*sharedMem1[ii];
					ii += warpSize; 
					y_n[1] += sharedMem2[(warpIdx<<5) + i]*sharedMem1[ii];
					ii += warpSize; 
					y_n[2] += sharedMem2[(warpIdx<<5) + i]*sharedMem1[ii];
					y_d += sharedMem2[(warpIdx<<5) + i];
				}
			}
			__syncwarp();
			sharedMem3[threadIdx.x] = 0;
			sharedMem2[threadIdx.x] = 0;
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


