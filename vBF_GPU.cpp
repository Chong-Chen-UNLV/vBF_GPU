#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vBF_GPU.h"

static void examineBF(float* inputImage, 
		float* outputImage,
		const int32_t rowC,
		const int32_t colC,
		const int32_t imageWidth,
		const int32_t imageHeight,
		const uint32_t numOfSpectral,
		const uint32_t windowSize,
		float sigmaInvR,
		float sigmaInvD){

	float y_n[numOfSpectral] = 0;	
	float y_d = 0;
	for(int32_t row = max(0, rowC - windowSize); row < min(imageHeight, rowC + windowSize); ++row){
		for(int32_t col = max(0, colC - windowSize); col < min(imageWidth, colC + windowSize); ++col){
			for(uint32_t level = 0; level < numOfSpectral; ++level){
				delta += pow(inputImage[(row*imageWidth + col)*numOfSpectral + level] - 
						inputImage[(row*imageWidth + col)*numOfSpectral + level], 2);
			}
			omega = exp(-delta*sigmaInvR - (pow(rowC - row, 2) + pow(colC - col, 2))*sigmaInvD);
			y_d += omega;
			for(uint32_t level = 0; level < numOfSpectral; ++level){
				y_n[level] += omega*inputImage[(row*imageWidth + col)*numOfSpectral + level];
			}
		}
		for(uint32_t level = 0; level < numOfSpectral; ++level){
			outputImage[(rowC*imageWidth + colC)*numOfSpectral + level] = y_n[level]/y_d;
		}
	}	

}
static void BF_CPU(float* inputImage, float* outputImage, 
		 const uint32_t imageWidth,
		 const uint32_t imageHeight, const uint32_t numOfSpectral,
		 const uint32_t windowSize, const float sigmaInvR, 
		 const float sigmaInvD){

	for(int32_t row = rowStart; row < rowEnd; ++row){
		for(int32_t col = 0; col < imageWidth; ++col){
			examineBF(inputImage, outputImage,
				row, col, imageWidth, imageHeight,
				numOfSpectral, windowSize, 
				sigmaInvR,sigmaInvD);
		}	
	}
}

static void BF_CPU8T(float* inputImage, float* outputImage, 
		 const uint32_t imageWidth,
		 const uint32_t imageHeight, const uint32_t numOfSpectral,
		 const uint32_t windowSize, const float sigmaInvR, 
		 const float sigmaInvD){
	
	const int32_t stride = imageHeight/8;
	rank = ;
	int32_t rowStart = rank*stride;
	int32_t rowEnd = min(imageHeight, rowStart + stride);
	for(int32_t row = rowStart; row < rowEnd; ++row){
		for(int32_t col = 0; col < imageWidth; ++col){
			examineBF(inputImage, outputImage,
					row, col, imageWidth, imageHeight,
					numOfSpectral, windowSize, 
					sigmaInvR,sigmaInvD);

		}	
	}
}

int main(int argc, char **argv){

	uint32_t imageWidth, imageHeight, windowSize, numOfSpectral;
	char inputImageFile[100];
	char outputImageFile[100];

	while ((oc = getopt(argc, argv, "i:o:")) != -1) {
		switch (oc) {
			case 'i':
				/* input image file*/
				sprintf(inputImageFile, "../images/%s.txt", optarg);		
				printf("input filename is %s\n", inputImageFile);
				break;
			case 'o':
				/* the number of cycles */
				sprintf(outputImageFile, "../images/%s.txt", optarg);		
				printf("output filename is %s\n", outputImageFile);
				break;
			case '?':
				printf("unrecongnized option\n");
				break;
			default:
				printf("option/arguments error!\n");       /* error handling, see text */
				exit(0);
		}
	}

	float *inputImage, *outputImage;
	float sigma_r=10, sigma_d=30.5;

	//read file
	FILE *fp1=fopen(inputImageFile,"r");
	fscanf(f, "%d %d %d", &imageWidth, &imageHeight, &numOfSpectral);
	uint32_t sizeI = imageWidth*imageHeight*numOfSpectral;
	uint32_t imageIdx;	
	inputImage = (float *)malloc(sizeI*sizeof(float));
	outputImage =(float *)malloc(sizeI*sizeof(float));
	
	float val;
	uint32_t row, col, level;
	
	//please notice that the image txt file is stacked by layers i.e. I(i, j, k) address is I[k*imageSize+i*row+j]
	//we reorder the order to make it more friendly for memory coalescing
	for(uint32_t i=0; i < sizeI; i++){
		fscanf(fp1,"%f",&val);
		level = i/(imageWidth*imageHeight);
		row = (i - (level*imageWidth*imageHeight))/imageWidth;
		col = (i - (level*imageWidth*imageHeight))%imageWidth;
		I[(row*imageWidth + col)*level + level]=val;
	}
	printf("finish read I\n");
	fclose(fp1);	

	struct timeval start1, end1;
	float timeByMs;
	float *B_d,*I_d;
	//-------------naive CPU runtime------------
	gettimeofday(&start1, NULL);
	BF_CPU(inputImage, outputImage, 
			imageWidth, imageHeight, numOfSpectral, 
			windowSize, sigmaR, sigmaD);
	gettimeofday(&end1, NULL);
	timeByMs=((end1.tv_sec * 1000000 + end1.tv_usec)-(start1.tv_sec * 1000000 + start1.tv_usec))/1000;	
	printf("time cost for CPU is %f ms\n",timeByMs);
	for(int i = 0; i < 20; i++){
		printf("test sample CPU %i is %f\n", outputImage[i*1000+2i*200+i*sizeI]);
	}
	//-------------8 thread CPU runtime----------------
	gettimeofday(&start1, NULL);
	BF_CPU8T(inputImage, outputImage, 
			imageWidth, imageHeight, numOfSpectral, 
			windowSize, sigmaR, sigmaD);
	gettimeofday(&end1, NULL);
	timeByMs=((end1.tv_sec * 1000000 + end1.tv_usec)-(start1.tv_sec * 1000000 + start1.tv_usec))/1000;	
	printf("time cost for 8 thread CPU is %f ms\n",timeByMs);
	for(int i = 0; i < 20; i++){
		printf("test sample CPU %i is %f\n", outputImage[i*1000+2i*200+i*sizeI]);
	}

	//------------start GPU testing----------

	cudaMalloc((void **) &inputImage_d, sizeI*sizeof(float));
	cudaMalloc((void **) &outputImage_d, sizeI*sizeof(float));
	
	//-------------naive GPU testing------------	
	gettimeofday(&start1, NULL);
	cudaMemcpy(inputImage_d, inputImage, sizeI*sizeof(float), cudaMemcpyHostToDevice);
	naiveGPU(inputImage, outputImage, 
			imageWidth, imageHeight, numOfSpectral, 
			windowSize, sigmaR, sigmaD);
	cudaMemcpy(outputImage, outputImage_d, sizeI*sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	gettimeofday(&end1, NULL);
	float timeByMs;
	timeByMs=((end1.tv_sec * 1000000 + end1.tv_usec)-(start1.tv_sec * 1000000 + start1.tv_usec))/1000;	
	printf("time cost for naiveGPU is %f ms\n",timeByMs);
	for(int i = 0; i < 20; i++){
		printf("test sample naive %i is %f\n", outputImage[i*1000+2i*200+i*sizeI]);
	}

	//------------vBF_GPU testing------------	
	gettimeofday(&start1, NULL);
	cudaMemcpy(inputImage_d, inputImage, sizeI*sizeof(float), cudaMemcpyHostToDevice);
	vBF_GPU(inputImage, outputImage, 
			imageWidth, imageHeight, numOfSpectral, 
			windowSize, sigmaR, sigmaD);
	cudaMemcpy(outputImage, outputImage_d, sizeI*sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	gettimeofday(&end1, NULL);
	float timeByMs;
	timeByMs=((end1.tv_sec * 1000000 + end1.tv_usec)-(start1.tv_sec * 1000000 + start1.tv_usec))/1000;	
	printf("time cost for deblur is %f ms\n",timeByMs);
	for(int i = 0; i < 20; i++){
		printf("test sample vBF_GPU %i is %f\n", outputImage[i*1000+2i*200+i*sizeI]);
	}
	
	//--------------------write part of results to file, you can change it to write the whole result-----------
	FILE *fp2 = fopen(outputImageFile, "w");
	int k=0;
	for(long i=0;i<sizeI;i+=100){
		fprintf(fp2,"%e\t", outputImage[i]);
		if (k==100){
			k=0;
			fprintf(fp2,"\n");
		} 
		k++;
	}
	fclose(fp2);

	return 1;
}

