#include <iostream>
#include <cstdlib>

using namespace std;

void createMatrix(int* myMat, int row, int col)
{
	for (int i = 0; i<row; i++)
	{
		for (int j = 0; j<col; j++)
		{
			myMat[i*col + j] = rand() % 10;
		}
	}
}

void printMatrix(int* myMat, int row, int col)
{
	for (int i = 0; i<row; i++)
	{
		for (int j = 0; j<col; j++)
		{
			cout << myMat[i*col + j] << " ";
		}
		cout << endl;
	}
}

__global__
void multiplyMatrix(int* matA, int* matB, int* resultMat, int rowA, int rowB, int colB)
{
	// The kernel launched is 2D, so we need to identify both the row and column that we want to compute
	int row = threadIdx.y;
	int column = threadIdx.x;
	int idx = row*blockDim.x + column;
	if (row < rowA && column < colB)
	{
		float val = 0;
		for (int i = 0; i < rowB; i++)
		{
			//rowB is the number of column in A
			val += matA[row*rowB + i] * matB[i*colB + column];
		}
		resultMat[idx] = val;
	}
}

void multiplyMatrixSerial(int* matA, int* matB, int* resultMat, int rowA, int rowB, int colB)
{
	for (int i = 0; i<rowA; i++)
	{
		for (int j = 0; j<colB; j++)
		{
			int sum = 0;
			for (int k = 0; k<rowB; k++)
			{
				sum += (matA[i*rowB + k] * matB[k*colB + j]);
			}
			resultMat[i*colB + j] = sum;
		}
	}
}

int main()
{
	int rowA = 3;
	int rowB = 4;
	int colB = 3;

	//allocate memory in host
	int *matA = new int[rowA*rowB * sizeof(int)];
	int *matB = new int[rowB*colB * sizeof(int)];
	int *matC = new int[rowA*colB * sizeof(int)];

	//allocate memory in device
	int *dA, *dB, *dC;
	cudaMalloc((void**)&dA, rowA*rowB * sizeof(int));
	cudaMalloc((void**)&dB, rowB*colB * sizeof(int));
	cudaMalloc((void**)&dC, rowA*colB * sizeof(int));

	cout << "Creating matrix..." << endl;
	createMatrix(matA, rowA, rowB);
	createMatrix(matB, rowB, colB);
	cout << "Creating matrix completed" << endl;

	//copy from host to device
	cudaMemcpy(dA, matA, rowA*rowB * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, matB, rowB*colB * sizeof(int), cudaMemcpyHostToDevice);

	cout << "MatrixA: " << endl;
	printMatrix(matA, rowA, rowB);
	cout << "MatrixB: " << endl;
	printMatrix(matB, rowB, colB);

	
	dim3 blockDim3(rowA, colB, 1);
	//Our grid will have (1,1,1) blocks while each block will have (rowA,colB,1) threads in x-,y- and z- directions
	//each thread will compute a single element in the resultant matrix
	multiplyMatrix << <1, blockDim3 >> > (dA, dB, dC, rowA, rowB, colB);

	//copy result from device to host
	cudaMemcpy(matC, dC, rowA*colB * sizeof(int), cudaMemcpyDeviceToHost);

	cout << "The parallel result matrix is: " << endl;
	printMatrix(matC, rowA, colB);

	multiplyMatrixSerial(matA, matB, matC, rowA, rowB, colB);
	cout << "The serial result matrix is: " << endl;
	printMatrix(matC, rowA, colB);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	delete[] matA;
	delete[] matB;
	delete[] matC;

	return 0;
}