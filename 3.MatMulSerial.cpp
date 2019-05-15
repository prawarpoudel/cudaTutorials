#include <iostream>
#include <cstdlib>

using namespace std;

void createMatrix(int* myMat,int row,int col)
{
	for(int i=0;i<row;i++)
	{
		for(int j=0;j<col;j++)
		{
			myMat[i*col+j] = rand()%10;
		}
	}
}

void printMatrix(int* myMat,int row,int col)
{
	for(int i=0;i<row;i++)
	{
		for(int j=0;j<col;j++)
		{
			cout<<myMat[i*col+j]<<" ";
		}
		cout<<endl;
	}
}

void multiplyMatrix(int* matA,int* matB,int* resultMat,int rowA,int rowB,int colB)
{
	for(int i=0;i<rowA;i++)
	{
		for(int j=0;j<colB;j++)
		{
			int sum = 0;
			for(int k=0;k<rowB;k++)
			{
				sum+=(matA[i*rowB+k]*matB[k*colB+j]);
			}
			resultMat[i*colB+j] = sum;
		}
	}
}

int main()
{
	int rowA = 3;
	int rowB = 4;
	int colB = 3;

	int *matA = new int[rowA*rowB*sizeof(int)];
	int *matB = new int[rowB*colB*sizeof(int)];
	int *matC = new int[rowA*colB*sizeof(int)];

	cout<<"Creating matrix..."<<endl;
	createMatrix(matA,rowA,rowB);
	createMatrix(matB,rowB,colB);
	cout<<"Creating matrix completed"<<endl;

	cout<<"MatrixA: "<<endl;
	printMatrix(matA,rowA,rowB);
	cout<<"MatrixB: "<<endl;
	printMatrix(matB,rowB,colB);

	multiplyMatrix(matA,matB,matC,rowA,rowB,colB);
	cout<<"The result matrix is: "<<endl;
	printMatrix(matC,rowA,colB);

	delete[] matA;
	delete[] matB;
	delete[] matC;

	return 0;
}