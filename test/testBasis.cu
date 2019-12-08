#include <stdio.h> 
#include "../DG_Basis.cuh"
#include "../DG_Quad.cuh"
#include "../CUDA_Helper.cuh"

int main()
{
  DG_BasisData *BasisData; 
  createBasisData(&BasisData);
  int p = 3;
  computeBasisData(p, BasisData);
  int order, np, nq1, nq2, i, j, edge;
  order = BasisData->order;
  np = BasisData->np; 
  nq1 = BasisData->nq1;
  nq2 = BasisData->nq2; 
  // inter info 
  printf("order %d # of basis fncs %d nq1 %d nq2 %d \n", order,np, nq1, nq2);
  /* interior quad points */
  // phi 
  double sum; 
  for (i=0; i<nq2; i++){
    sum = 0.0; 
    printf("quadpoint %d: %f %f \n", i, BasisData->xyq[2*i], BasisData->xyq[2*i+1]);
    for (j=0; j<np; j++) {printf("jth basis fncs %d: %.10f\n", j, BasisData->Phi[i*np+j]); sum+=BasisData->Phi[i*np+j];}
    printf("sum is: %f \n", sum);
  }
  printf("**************************************************\n" );


  // Gphix 
  for (i=0; i<nq2; i++){
    sum = 0.0; 
    printf("quadpoint %d: %f %f \n", i, BasisData->xyq[2*i], BasisData->xyq[2*i+1]);
    for (j=0; j<np; j++) {printf("jth Gradx basis fncs %d: %f\n", j, BasisData->GPhix[i*np+j]); sum += BasisData->GPhix[i*np+j];}
    printf("sum is: %f \n", sum);
  }
  printf("**************************************************\n" );
  //Gphiy 
  for (i=0; i<nq2; i++){
    sum = 0.0;
    printf("quadpoint %d: %f %f \n", i, BasisData->xyq[2*i], BasisData->xyq[2*i+1]);
    for (j=0; j<np; j++) {printf("jth Grady basis fncs %d: %f\n", j, BasisData->GPhiy[i*np+j]); sum += BasisData->GPhiy[i*np+j]; }
    printf("sum is: %f \n", sum);
  }
  
  for (edge=0; edge<3; edge++)
  {
    printf("********************************\n");
    printf("edge %d quad info \n",edge);
    for (i=0; i<nq1; i++)
    {
      sum = 0; 
      printf("%dth quad point: %.8f\n", i, BasisData->sq[i]);
      for (j=0; j<np; j++) {
        printf("%.8f ", BasisData->EdgePhiL[edge][i*np+j]);
        sum +=  BasisData->EdgePhiL[edge][i*np+j];
      }
      printf("sum is: %f \n", sum);
      printf("Right *******\n");
      sum = 0.0;
      for (j=0; j<np; j++) {
        printf("%.8f ", BasisData->EdgePhiR[edge][(nq1-1-i)*np+j]);
        sum +=  BasisData->EdgePhiR[edge][i*np+j];
      }
      printf("sum is: %f \n", sum);
    }

  }

  freeBasisData(BasisData);
  CUDA_CALL(cudaDeviceReset());
  return 0;
}
