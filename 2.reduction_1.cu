#include <cuda.h>
#include <stdio.h>

__device__ 
int ceilDiv(int num1,int num2)
{
	int adder = 0;
	if(num1%num2) adder = 1;

	return ((int)num1/num2)+adder;
}

__global__
void reduceVec(int* vecA,int* answer,int size)
{
	int i = threadIdx.x;
	int halfSize = ceilDiv(size,2);
	int prevSize = size;

	while(halfSize>1)
	{
		if(i<halfSize && (i+halfSize)<prevSize)
		{
			vecA[i] += vecA[i+halfSize];
		}
		__syncthreads();
		prevSize = halfSize;
		halfSize = ceilDiv(halfSize,2);
	}
	__syncthreads();
	if(i==0) 
	{
		if(size>1)	*answer = vecA[0]+vecA[1];
		else if (size) *answer = vecA[0];
		else *answer = 0;
	}
}

int main(int argc,char** argv)
{
	int SIZEARRAY = 0;
	if(argc==2)
	{
		SIZEARRAY = atoi(argv[1]);
	}else
	{
		SIZEARRAY = 400;
	}

	//allocate memory in host
	int* myVec = new int[SIZEARRAY];
	int* result = new int;
	// *result = 700;

	//allocate memory in the device
	int *dVec,*dResult;
	cudaMalloc((void**)&dVec,SIZEARRAY*sizeof(int));
	cudaMalloc((void**)&dResult,sizeof(int));

	for(int i=0;i<SIZEARRAY;i++)
	{
		myVec[i] = 1;
	}
	//copy data to device
	cudaMemcpy(dVec,myVec,SIZEARRAY*sizeof(int),cudaMemcpyHostToDevice);

	//call the kernel
	reduceVec<<<1,SIZEARRAY>>>(dVec,dResult,SIZEARRAY);

	//copy result to the host
	cudaMemcpy(result,dResult,sizeof(int),cudaMemcpyDeviceToHost);

	//display result
	printf("The sum is %d\n",*result);

	cudaFree(dVec);
	cudaFree(dResult);

	delete[] myVec;
	delete result;
	return 0;
}