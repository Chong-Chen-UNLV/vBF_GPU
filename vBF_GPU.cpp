#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "vBF_GPU.h"

static void examineBF(float* inputImage, 
		float* outputImage,
		const int32_t rowC,
		const int32_t colC,
		const int16_t windowSize,
		float sigmaInvR,
		float sigmaInvD){
	
	float y_n[numOfSpectral] = {0};	
	float y_d = 0;
	float delta = 0;
	float omega = 0;
	int32_t rowStart = std::max(0, rowC -(int)windowSize);
	int32_t rowEnd = std::min((int)imageHeight - 1 , rowC + (int)windowSize);
	int32_t colStart = std::max(0, colC - (int)windowSize);
	int32_t colEnd = std::min((int)imageWidth - 1, colC + (int)windowSize);
	int iter = 0;
	FILE* ft;
	if(rowC == 90 && colC == 60)
		ft = fopen("row90col60.txt", "w");
	for(int32_t row = rowStart; row <= rowEnd; ++row){
		for(int32_t col = colStart; col <= colEnd; ++col){
			delta = 0;
			for(uint32_t level = 0; level < numOfSpectral; ++level){
				delta += pow(inputImage[(row*imageWidth + col)*numOfSpectral + level] - 
						inputImage[(rowC*imageWidth + colC)*numOfSpectral + level], 2);
			}
			omega = exp(-delta*sigmaInvR - (pow(rowC - row, 2) + pow(colC - col, 2))*sigmaInvD);
			y_d += omega;
			if(rowC == 90 && colC == 60)
				fprintf(ft, "at %d th iter with row = %d col = %d omega is %f, y_d is %f\n", iter, row, col, delta, y_d);
						
			iter += 1;
			for(uint32_t level = 0; level < numOfSpectral; ++level){
				y_n[level] += omega*inputImage[(row*imageWidth + col)*numOfSpectral + level];
			}
		}
	}	
	for(uint32_t level = 0; level < numOfSpectral; ++level){
		outputImage[(rowC*imageWidth + colC)*numOfSpectral + level] = y_n[level]/y_d;
	}
	if(rowC == 90 && colC == 60)
		fclose(ft);
}
static void BF_CPU(float* inputImage, float* outputImage, 
		 const int16_t windowSize, const float sigmaR, 
		 const float sigmaD){

	float sigmaInvR = .5/pow(sigmaR, 2);
	float sigmaInvD = .5/pow(sigmaD, 2);
	for(int32_t row = 0; row < 1; ++row){
		for(int32_t col = 0; col < imageWidth; ++col){
			examineBF(inputImage, outputImage,
				row, col, windowSize, 
				sigmaInvR,sigmaInvD);
		}	
		//printf("row %d finished\n", row);
	}
}

static void BF_CPU8T(float* inputImage, float* outputImage, 
		 const uint16_t windowSize, const float sigmaR, 
		 const float sigmaD){
	
	const int32_t stride = imageHeight/8;
	int rank;
	float sigmaInvR = .5/pow(sigmaR, 2);
	float sigmaInvD = .5/pow(sigmaD, 2);
	omp_set_num_threads(8);
	#pragma omp parallel private(rank)
	{
		rank = omp_get_thread_num();
		int32_t rowStart = rank*stride;
		int32_t rowEnd = std::min((int)imageHeight, rowStart + stride);
		for(int32_t row = rowStart; row < rowEnd; ++row){
			for(int32_t col = 0; col < imageWidth; ++col){
				examineBF(inputImage, outputImage,
						row, col, windowSize, 
						sigmaInvR,sigmaInvD);
			}	
		}
	}
}

int main(int argc, char **argv)
{

	int32_t imageWidth_, imageHeight_, numOfSpectral_;
	int16_t windowSize; 
	char inputImageFile[100];
	char outputImageFile[100];
	int oc;
	while ((oc = getopt(argc, argv, "i:o:w:")) != -1) {
		switch (oc) {
			case 'i':
				/* input image file*/
				sprintf(inputImageFile, "./%s.txt", optarg);		
				printf("input filename is %s\n", inputImageFile); break;
			case 'o':
				/* the number of cycles */
				sprintf(outputImageFile, "./%s.txt", optarg);		
				printf("output filename is %s\n", outputImageFile);
				break;
			case 'w':
				windowSize = atoi(optarg);
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
	float *inputImage2, *outputImage2;
	float sigmaR = 15, sigmaD=20.5;

	//read file
	FILE *fp1=fopen(inputImageFile,"r");
	fscanf(fp1, "%d %d %d", &imageWidth_, &imageHeight_, &numOfSpectral_);
	if(imageHeight_ != imageHeight || imageWidth_ != imageWidth || numOfSpectral_ != numOfSpectral)
		exit(0);
	uint32_t sizeI = imageWidth*imageHeight*numOfSpectral;
	inputImage = (float *)malloc(sizeI*sizeof(float));
	inputImage2 = (float *)malloc(sizeI*sizeof(float));
	outputImage =(float *)calloc(sizeI, sizeof(float));
	outputImage2 =(float *)calloc(sizeI, sizeof(float));
	
	float val;
	uint32_t row, col, level;
	
	//please notice that the image txt file is stacked by layers i.e. I(i, j, k) address is I[k*imageSize+i*row+j]
	//we reorder the order to make it more friendly for memory coalescing
	for(uint32_t i=0; i < sizeI; i++){
		fscanf(fp1,"%f",&val);
		val = val/4096;
		level = i/(imageWidth*imageHeight);
		row = (i - (level*imageWidth*imageHeight))/imageWidth;
		col = (i - (level*imageWidth*imageHeight))%imageWidth;
		inputImage[(row*imageWidth + col)*numOfSpectral + level]=val;
		if(level*imageHeight*imageWidth+row*imageWidth+col != i)
			printf("i error \n");
		inputImage2[i]=val;
	}
	printf("finish read I\n");
	fclose(fp1);	

	struct timeval start1, end1;
	float timeByMs;
	//-------------naive CPU runtime------------
	//gettimeofday(&start1, NULL);
	//BF_CPU(inputImage, outputImage, 
	//		windowSize, sigmaR, sigmaD);
	//gettimeofday(&end1, NULL);
	//timeByMs=((end1.tv_sec * 1000000 + end1.tv_usec)-(start1.tv_sec * 1000000 + start1.tv_usec))/1000;	
	//printf("time cost for CPU is %f ms\n",timeByMs);
	//for(int i = 0; i < 5; i++){
	//	printf("test sample CPU %i is %f\n", i*1000+2*i*200+i*imageWidth*imageHeight, outputImage[i*1000+2*i*200+i*imageWidth*imageHeight]);
	//}
	//-------------8 thread CPU runtime----------------
	gettimeofday(&start1, NULL);
	BF_CPU8T(inputImage, outputImage, 
			windowSize, sigmaR, sigmaD);
	gettimeofday(&end1, NULL);
	timeByMs=((end1.tv_sec * 1000000 + end1.tv_usec)-(start1.tv_sec * 1000000 + start1.tv_usec))/1000;	
	printf("time cost for 8 thread CPU is %f ms\n",timeByMs);
	for(int i = 0; i < 5; i++){
		printf("test sample 8-thread CPU %d, %d at level %d is %f\n", 10*i*i,  20*i, i, outputImage[i + (10*i*i*imageWidth + 20*i)*numOfSpectral]);
	}

	//------------start GPU testing----------

	float *inputImage_d, *outputImage_d;
	cudaMalloc((void **) &inputImage_d, sizeI*sizeof(float));
	cudaMalloc((void **) &outputImage_d, sizeI*sizeof(float));
	
	//-------------naive GPU testing------------	
	gettimeofday(&start1, NULL);
	cudaMemcpy(inputImage_d, inputImage2, sizeI*sizeof(float), cudaMemcpyHostToDevice);
	naiveGPU(inputImage_d, outputImage_d, 
			windowSize, sigmaR, sigmaD);
	cudaMemcpy(outputImage2, outputImage_d, sizeI*sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	gettimeofday(&end1, NULL);
	timeByMs=((end1.tv_sec * 1000000 + end1.tv_usec)-(start1.tv_sec * 1000000 + start1.tv_usec))/1000;	
	printf("time cost for naiveGPU is %f ms\n",timeByMs);
	for(int i = 0; i < 5; i++){
		printf("test sample naive GPU %d, %d at level %d is %f\n", 10*i*i,  20*i, i, outputImage2[i*imageHeight*imageWidth + 10*i*i*imageWidth + 20*i]);
	}

	//------------vBF_GPU testing------------	
	gettimeofday(&start1, NULL);
	cudaMemcpy(inputImage_d, inputImage, sizeI*sizeof(float), cudaMemcpyHostToDevice);
	vBF_GPU(inputImage_d, outputImage_d, 
			windowSize, sigmaR, sigmaD);
	cudaMemcpy(outputImage, outputImage_d, sizeI*sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	gettimeofday(&end1, NULL);
	timeByMs=((end1.tv_sec * 1000000 + end1.tv_usec)-(start1.tv_sec * 1000000 + start1.tv_usec))/1000;	
	printf("time cost for deblur is %f ms\n",timeByMs);
	for(int i = 0; i < 5; i++){
		printf("test sample vBF_CPU %d, %d at level %d is %f\n", 10*i*i,  20*i, i, outputImage[i + (10*i*i*imageWidth + 20*i)*numOfSpectral]);
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

