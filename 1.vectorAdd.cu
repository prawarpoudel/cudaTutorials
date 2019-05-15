#include <cuda.h>
#include <stdio.h>

#define SIZEARRAY 200

__global__
void addVec(int* vecA, int* vecB, int* vecC, int size)
{
	int i = threadIdx.x;
	vecC[i] = vecA[i] + vecB[i];
}

void printArray(int* array)
{
	for (int i = 0; i<SIZEARRAY; i++)
	{
		printf("%d ", array[i]);
	}
}

int main()
{
	//allocating memory for host
	int *A = new int[SIZEARRAY];
	int *B = new int[SIZEARRAY];
	int *C = new int[SIZEARRAY];

	//allocating memory for device
	int *dA, *dB, *dC;
	cudaMalloc((void**)&dA, SIZEARRAY * sizeof(int));
	cudaMalloc((void**)&dB, SIZEARRAY * sizeof(int));
	cudaMalloc((void**)&dC, SIZEARRAY * sizeof(int));

	//initialize A and B in host
	for (int i = 0; i<SIZEARRAY; i++)
	{
		A[i] = i;
		B[i] = SIZEARRAY - i;
	}

	//copy from the host to device
	cudaMemcpy(dA, A, SIZEARRAY * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, SIZEARRAY * sizeof(int), cudaMemcpyHostToDevice);

	//call the kernel
	addVec << <1, SIZEARRAY >> >(dA, dB, dC, SIZEARRAY);

	//copy data from device to host
	cudaMemcpy(C, dC, SIZEARRAY * sizeof(int), cudaMemcpyDeviceToHost);

	//.. at this point the data will be available to the host machine

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	printArray(C);

	delete[] A;
	delete[] B;
	delete[] C;
}