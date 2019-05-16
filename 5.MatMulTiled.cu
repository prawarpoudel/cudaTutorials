#include <iostream>
#include <cstdlib>
#include <math.h>

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

__global__
void multiplyMatrixSharedMem(int* matA, int* matB, int* resultMat, int rowA, int rowB, int colB)
{
	//following are the idx related to this particular thread
	int bRow = blockIdx.y, bCol = blockIdx.x;
	int tRow = threadIdx.y, tCol = threadIdx.x;

	int answer = 0;

	//since the elements are shared by threads in a block, we create
	//..appropriate shared memory
	extern __shared__ int sharedMem[];

	int* valA = &sharedMem[0];
	int* valB = &sharedMem[blockDim.x*blockDim.y];

	//following are the indices of this thread in global scope
	int rIdx = blockDim.y*bRow + tRow;
	int cIdx = blockDim.x*bCol + tCol;

	//we need to go through all the tiles that we created in x- directions
	for (int ii = 0; ii < (rowA+blockDim.x-1)/blockDim.x; ii++)
	{
		//load value from fiorst matrix from global scope, in global score colm of A is row if B
		//.. rIdx*rowB+ii*blockDim.x helps to locate the appropriate tile while tCol will find the col in that tile
		//.. but since the code is designed to be work with rectangle matrix also, we have to check and make sure we are in bounds.
		if ((ii*blockDim.x + tCol < rowB) && (rIdx<rowA))
		{
			valA[tRow*blockDim.y+tCol] = matA[rIdx*rowB + ii*blockDim.x + tCol];
		}
		else
			//for out of bounds we will assign 0, and later ignore
			valA[tRow*blockDim.y + tCol] = 0;
		
		if ((ii*blockDim.y + tRow < rowB) && (cIdx<colB))
		{
			valB[tRow*blockDim.y + tCol] = matB[(ii*blockDim.y+tRow)*colB+cIdx];
		}
		else
			//for out of bounds we will assign 0, and later ignore
			valB[tRow*blockDim.y + tCol] = 0;
		__syncthreads();

		// now the value are loaded, we will perform the calculation,
		//.. remember the final value is not computed at one go, rather than after certain number of iterations
		for (int iii = 0; iii < blockDim.y; iii++ )
		{
			answer += valA[tRow*blockDim.y + iii] * valB[iii*blockDim.y+tCol];
		}
		__syncthreads();
	}
	
	if (cIdx < colB && rIdx < rowA)
		resultMat[rIdx*colB + cIdx] = answer;
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

bool checkMatrices(int* refMat, int* checkMat, int row, int col)
{
	for (int rid = 0; rid < row; rid++)
	{
		for (int cid = 0; cid < col; cid++)
		{
			if (refMat[rid*col + cid] != checkMat[rid*col + cid])
				return false;
		}
	}
	return true;
}

int main()
{
	int rowA = 300;
	int rowB = 400;
	int colB = 50;


	//allocate memory in host
	int *matA = new int[rowA*rowB * sizeof(int)];
	int *matB = new int[rowB*colB * sizeof(int)];
	int *matC = new int[rowA*colB * sizeof(int)];
	int *matD = new int[rowA*colB * sizeof(int)];
	
	//get the device properties to find the maximum number of threads that can go in a block
	cudaDeviceProp  prop;
	int count;
	cudaGetDeviceCount(&count);

	if (count <1)
	{
		cout << "no CUDA enabled device found" << endl;
		return -1;
	}
	//we wil only consider one GPU here, so 0th device
	cudaGetDeviceProperties(&prop, 0);
	int maxThreadInBlock = prop.maxThreadsPerBlock;

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

	int threadsX = sqrt(maxThreadInBlock);
	dim3 blockDim3(threadsX, threadsX, 1);

	//since the size of shared Memory should be a constant value
	//.. so we will define the size here
	int sharedMemorySize = threadsX* threadsX * sizeof(int) * 2;
	dim3 gridDim3((rowA+threadsX-1)/threadsX, (colB + threadsX - 1) / threadsX, 1);

	//Our grid will have (1,1,1) blocks while each block will have (rowA,colB,1) threads in x-,y- and z- directions 
	//each thread will compute a single element in the resultant matrix
	multiplyMatrixSharedMem << <gridDim3, blockDim3, sharedMemorySize >> > (dA, dB, dC, rowA, rowB, colB);

	//copy result from device to host
	cudaMemcpy(matC, dC, rowA*colB * sizeof(int), cudaMemcpyDeviceToHost);

	multiplyMatrixSerial(matA, matB, matD, rowA, rowB, colB);

	if (checkMatrices(matC, matD, rowA,colB))
		cout << "The matrix computation is incorrect" << endl;
	else
		cout << " The matrix computation is correct" << endl;

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	delete[] matA;
	delete[] matB;
	delete[] matC;
	delete[] matD;

	return 0;
}