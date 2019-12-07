#include <stdio.h>
#include <stdlib.h>
#include "../DG_Math.cuh"

void printMatrix(int m, int n, double *A){
  int i, j; 
  for (i=0; i<m; i++){
    for (j=0; j<n; j++)
      printf("%.5f ", A[i*n+j]);
    printf("\n"); 
  }

}
int main(int argc, char const *argv[])
{
  double A[3*2] = {1, 2, 3, 4, 5, 6};
  double AT[2*3] = {1,3,5,2,4,6};
  double B[2*3] = {7, 8, 9, 10, 11, 12}; 
  double BT[3*2] = {7,10,8,11,9,12}; 
  double C[9] = {0};
  double iA[9] = {0};
  double iB[4] = {0}; 

  printf("A = \n"); 
  printMatrix(3,2, A);
  printf("B = \n"); 
  printMatrix(2,3,B); 
  
  printf("C=AxB\n"); 
  DG_MxM_Set(3,2,3,A,B,C); 
  printMatrix(3,3,C); 
  
  printf("C=0.5*AxB\n"); 
  DG_cMxM_Set(0.5,3,2,3,A,B,C); 
  printMatrix(3,3,C);
  
  printf("C+=AxB\n");
  DG_MxM_Add(3,2,3,A,B,C); 
  printMatrix(3,3,C); 

  printf("C+=-AxB\n");
  DG_cMxM_Add(-1,3,2,3,A,B,C);
  printMatrix(3,3,C); 

  printf("C-=AxB\n");
  DG_MxM_Sub(3,2,3,A,B,C); 
  printMatrix(3,3,C); 

  printf("C-=-AxB\n");
  DG_cMxM_Sub(-1.0,3,2,3,A,B,C); 
  printMatrix(3,3,C);

  printf("C = A*AT\n");
  DG_MxMT_Set(3,2,3, A, A, C);
  printMatrix(3,3,C);
  printf("C = B*BT\n");
  DG_MxMT_Set(2,3,2, B, B, C);
  printMatrix(2,2,C);
  printf("C = ATxBT\n");
  DG_MxMT_Set(2,3,2, AT, B, C);
  printMatrix(2,2,C);
  printf("C = BTxAT\n");
  DG_MxMT_Set(3,2,3,BT, A, C);
  printMatrix(3,3,C);
  
  printf("C = ATxA\n");
  DG_MTxM_Set(2,3,2, A,A,C);
  printMatrix(2,2,C);
  printf("C = BTxB\n");
  DG_MTxM_Set(3,2,3,B,B,C);
  printMatrix(3,3,C);
  printf("C = ATxBT\n");
  DG_MTxM_Set(2,3,2,A,BT,C);
  printMatrix(2,2,C);
  printf("C = BTxAT\n");
  DG_MTxM_Set(3,2,3,B,AT,C);
  printMatrix(3,3,C);

  printf("C += BTxAT\n");
  DG_MTxM_Add(3,2,3,B,AT,C);
  printMatrix(3,3,C);
  printf("C += BTxB\n");
  DG_MTxM_Add(3,2,3,B,B, C);
  printMatrix(3,3,C);


  printf("test inversion\n");
  printf("iA = (A^TxA)^-1\n");
  DG_MTxM_Set(2,3,2,A,A,C);
  printMatrix(2,2,C);
  DG_Inv(2,C,iA);
  printMatrix(2,2,C);
  printMatrix(2,2,iA);
  printf("iB = (BxB^T)\n");
  DG_MxMT_Set(2,3,2,B,B,C);
  printMatrix(2,2,C);
  DG_Inv(2,C,iB);
  printMatrix(2,2,C);
  printMatrix(2,2,iB);

  return 0;
}
