#include <iostream>

#define SIZEARRAY 200

void addVec(int* vecA,int* vecB,int* vecC,int size)
{
	for(int i=0;i<size;i++)
	{
		vecC[i] = vecA[i]+vecB[i];
	}
}

int main()
{
	int *A = new int[SIZEARRAY];
	int *B = new int[SIZEARRAY];
	int *C = new int[SIZEARRAY];

	//initialize A and B
	for(int i=0;i<SIZEARRAY;i++)
	{
		A[i] = i;
		B[i] = SIZEARRAY-i;
	}

	addVec(A,B,C,SIZEARRAY);

	delete[] A;
	delete[] B;
	delete[] C;
}