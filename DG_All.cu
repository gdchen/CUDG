/* This is the source file for the All struct
 * 
 * Author: Guodong Chen
 * Email: cgderic@umich.edu
 * Last modified: 12/06/2019
 */ 

#include <stdlib.h>
#include "CUDA_Helper.cuh"
#include "DG_Mesh.cuh"
#include "DG_Basis.cuh"
#include "DG_DataSet.cuh"
#include "DG_All.cuh"

cudaError_t initAll(DG_All *All)
{
  All->Mesh = NULL;
  All->BasisData = NULL;
  All->DataSet = NULL;
  return cudaSuccess; 
}


cudaError_t createAll(DG_All **pAll)
{
  CUDA_CALL(cudaMallocManaged(pAll, sizeof(DG_All))); 
  initAll(*pAll);
  return cudaSuccess; 
}

// initialization all structures 
cudaError_t getAllFromIC(DG_All *All, int order, int N, double halfL) 
{
  // Mesh generation  
  CUDA_CALL(createMesh(&(All->Mesh)));
  CUDA_CALL(generateMesh(All->Mesh, halfL, N));
  CUDA_CALL(computeMeshInfo(All->Mesh));
  // Basis 
  CUDA_CALL(createBasisData(&(All->BasisData)));
  CUDA_CALL(computeBasisData(order,All->BasisData));
  // DataSet
  CUDA_CALL(createDataSet(&(All->DataSet)));
  CUDA_CALL(computeMassMatrix(All->DataSet, All->Mesh, All->BasisData));
  CUDA_CALL(interpolateIC(All->DataSet, All->Mesh));

  return cudaSuccess; 
}

cudaError_t freeAll(DG_All *All)
{
  CUDA_CALL(freeMesh(All->Mesh));
  CUDA_CALL(freeBasisData(All->BasisData));
  CUDA_CALL(freeDataSet(All->DataSet));
  CUDA_CALL(cudaFree(All)); 
  return cudaSuccess;
}

