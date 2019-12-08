#include <stdio.h>
#include "../DG_All.cuh"
#include "../DG_Mesh.cuh"
#include "../DG_Basis.cuh"
#include "../DG_DataSet.cuh"
//#include "DG_Residual.cuh"
//#include "DG_PostProcess.cuh"




int main(int argc, char const *argv[])
{
  
  int order = 1;
  int N = 16; 
  double halfL = 5;
  DG_All *All;
  createAll(&All);
  getAllFromIC(All, order, N, halfL);
  //int nElem = All->Mesh->nElem;
  //int np = All->BasisData->np;
  //int n, i, j;
  //printf("%d %d %d \n", nElem, All->BasisData->nq1, All->BasisData->nq2);
//  double dt = 0.001; 
//  //int Nt = 1;
//  int Nt = 5414;
//  //int Nt = 14142;
//  DG_Solve(All, dt, Nt);
//  //writeStates(All);
//  //writeStatesP(All,4);
//  double eu, es, ep;
//  ErrEst(All, &eu, &es, &ep);
//  printf("%f %f %f \n", sqrt(eu/25), sqrt(es/25), ep/25);
  freeAll(All); 
  CUDA_CALL(cudaDeviceReset());
  return 0;
}
