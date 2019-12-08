/* This is the source file contains the methods for the DG_Basis structs
 *
 * Author: Guodong Chen
 * Email: cgderic@umich.edu
 * Last modified: 12/05/2019
 */ 

#include "stdlib.h"
#include "CUDA_Helper.cuh"
#include "DG_Quad.cuh"
#include "DG_Basis.cuh"


/* initialize the Basis struct */ 
cudaError_t initBasisData(DG_BasisData *BasisData)
{
  BasisData->order = 0;
  BasisData->np = 0;
  
  BasisData->nq1 = 0;
  BasisData->sq = NULL;
  BasisData->wq1 = NULL;
  BasisData->EdgePhiL = NULL;
  BasisData->EdgePhiR = NULL;

  BasisData->nq2 = 0;
  BasisData->xyq = NULL;
  BasisData->wq2 = NULL;
  BasisData->Phi = NULL;
  BasisData->GPhix = NULL;
  BasisData->GPhiy = NULL;  
  
  return cudaSuccess; 
}


/* allocates the initialize the DG_Basis */ 
cudaError_t createBasisData(DG_BasisData **pBasisData)
{

  CUDA_CALL(cudaMallocManaged(pBasisData, sizeof(DG_BasisData))); 
  CUDA_CALL(initBasisData(*pBasisData));
  return cudaSuccess; 
}



/* free the memory of the Basis Data */
cudaError_t freeBasisData(DG_BasisData *BasisData)
{
  CUDA_CALL(cudaFree(BasisData->sq)); 
  CUDA_CALL(cudaFree(BasisData->wq1));
  CUDA_CALL(cudaFree(BasisData->xyq));  
  CUDA_CALL(cudaFree(BasisData->wq2)); 
  CUDA_CALL(cudaFree(BasisData->Phi)); 
  CUDA_CALL(cudaFree(BasisData->GPhix)); 
  CUDA_CALL(cudaFree(BasisData->GPhiy));

  CUDA_CALL(cudaFree(BasisData->EdgePhiL[0]));
  CUDA_CALL(cudaFree(BasisData->EdgePhiR[0]));
  CUDA_CALL(cudaFree(BasisData->EdgePhiL));
  CUDA_CALL(cudaFree(BasisData->EdgePhiR));
  CUDA_CALL(cudaFree(BasisData));

  return cudaSuccess; 
}


/* comupte the Basis data, allocate and fill in the members of BasisData */
cudaError_t computeBasisData(int p, DG_BasisData *BasisData)
{
  BasisData->order = p;            // Basis order
  BasisData->np = (p+1)*(p+2)/2;   // number of degrees of freedom, np 
  // Get Edge Quad Points, integrates up to order 2p+1 
  CUDA_CALL(DG_QuadLine(2*p+1, &(BasisData->nq1), &(BasisData->sq), &(BasisData->wq1)));
  // Get 2d Quad Points, integrates up to order 2p+1 
  CUDA_CALL(DG_QuadTriangle(2*p+1, &(BasisData->nq2), &(BasisData->xyq), &(BasisData->wq2)));

  int i, j, edge, np, nq1, nq2;
  // get the gradients of the basis function 
  np = BasisData->np;
  nq1 = BasisData->nq1;
  nq2 = BasisData->nq2;
  double xy[2];
  double GPhi[2*np]; 
  CUDA_CALL(cudaMallocManaged(&(BasisData->Phi), nq2*np*sizeof(double)));
  CUDA_CALL(cudaMallocManaged(&(BasisData->GPhix), nq2*np*sizeof(double)));
  CUDA_CALL(cudaMallocManaged(&(BasisData->GPhiy), nq2*np*sizeof(double))); 
  // evaluate the Phi and Gphi at the quad points 
  for (i=0; i<nq2; i++){
    xy[0] = BasisData->xyq[2*i];
    xy[1] = BasisData->xyq[2*i+1];
    DG_TriLagrange(p, xy, BasisData->Phi+i*np);
    DG_Grad_TriLagrange(p, xy, GPhi);
    for (j=0; j<np; j++){
      BasisData->GPhix[i*np+j] = GPhi[j];
      BasisData->GPhiy[i*np+j] = GPhi[np+j];
    }
  }
  // the edge basis for left and right element 
  double *tempL, *tempR; 
  CUDA_CALL(cudaMallocManaged(&(BasisData->EdgePhiL), 3*sizeof(double *))); 
  CUDA_CALL(cudaMallocManaged(&tempL, 3*nq1*np*sizeof(double)));
  CUDA_CALL(cudaMallocManaged(&(BasisData->EdgePhiR), 3*sizeof(double *))); 
  CUDA_CALL(cudaMallocManaged(&tempR, 3*nq1*np*sizeof(double)));
  // evaluate the basis at the egde nodes 
  for (edge=0; edge<3; edge++){
    BasisData->EdgePhiL[edge] = tempL + edge*nq1*np; 
    BasisData->EdgePhiR[edge] = tempR + edge*nq1*np; 
    for (i=0; i<nq1; i++){
      RefEdge2Elem(edge, xy, BasisData->sq[i]);
      DG_TriLagrange(p, xy, BasisData->EdgePhiL[edge]+i*np);
      RefEdge2Elem(edge, xy, 1.0-BasisData->sq[i]);
      DG_TriLagrange(p, xy, BasisData->EdgePhiR[edge]+i*np);
    }

  }

  return cudaSuccess; 

}



/* map 1d edge nodes to elem 2d coords */ 
cudaError_t RefEdge2Elem(const int edge, double *xy, const double sq) 
{
  switch (edge){
    case 0:
      xy[0] = 1.0-sq;
      xy[1] = sq;
      break;
    case 1:
      xy[0] = 0.0;
      xy[1] = 1.0-sq;
      break;
    case 2:
      xy[0] = sq;
      xy[1] = 0.0;
      break;
    default:
      return cudaErrorNotSupported;
      break;
  }
  return cudaSuccess;
}

/* evaluate basis function at xy */ 
cudaError_t DG_TriLagrange(int p, const double *xy, double *phi)
{
  double x, y;

  x = xy[0];
  y = xy[1];

  switch (p) {
    
  case 0:
    phi[0] = 1.0;
    return cudaSuccess;
    break;

  case 1:
    phi[0] = 1-x-y;
    phi[1] =   x  ;
    phi[2] =     y;
    return cudaSuccess;
    break;

  case 2:
    phi[0] = 1.0-3.0*x-3.0*y+2.0*x*x+4.0*x*y+2.0*y*y;
    phi[2] = -x+2.0*x*x;
    phi[5] = -y+2.0*y*y;
    phi[4] = 4.0*x*y;
    phi[3] = 4.0*y-4.0*x*y-4.0*y*y;
    phi[1] = 4.0*x-4.0*x*x-4.0*x*y;
    return cudaSuccess;
    break;
     
  case 3:
    phi[0] = 1.0-11.0/2.0*x-11.0/2.0*y+9.0*x*x+18.0*x*y+9.0*y*y-9.0/2.0*x*x*x-27.0/2.0*x*x*y-27.0/2.0*x*y*y-9.0/2.0*y*y*y;
    phi[3] = x-9.0/2.0*x*x+9.0/2.0*x*x*x;
    phi[9] = y-9.0/2.0*y*y+9.0/2.0*y*y*y;
    phi[6] = -9.0/2.0*x*y+27.0/2.0*x*x*y;
    phi[8] = -9.0/2.0*x*y+27.0/2.0*x*y*y;
    phi[7] = -9.0/2.0*y+9.0/2.0*x*y+18.0*y*y-27.0/2.0*x*y*y-27.0/2.0*y*y*y;
    phi[4] = 9.0*y-45.0/2.0*x*y-45.0/2.0*y*y+27.0/2.0*x*x*y+27.0*x*y*y+27.0/2.0*y*y*y;
    phi[1] = 9.0*x-45.0/2.0*x*x-45.0/2.0*x*y+27.0/2.0*x*x*x+27.0*x*x*y+27.0/2.0*x*y*y;
    phi[2] = -9.0/2.0*x+18.0*x*x+9.0/2.0*x*y-27.0/2.0*x*x*x-27.0/2.0*x*x*y;
    phi[5] = 27.0*x*y-27.0*x*x*y-27.0*x*y*y;
    return cudaSuccess;
    break;

  default:
    return cudaErrorNotSupported;
    break;

  }
  return cudaSuccess;
}




// gradients of basis functions at reference elements
cudaError_t DG_Grad_TriLagrange(int p, const double *xy, double *gphi)
{ 
  double x, y;
  int n = (p+1)*(p+2)/2;
  
  x = xy[0];
  y = xy[1];

  switch (p){
    
  case 0:
    gphi[0] =  0.0;
    gphi[n+0] =  0.0; 
    break;

  case 1:
    gphi[0] =  -1.0;
    gphi[1] =  1.0;
    gphi[2] =  0.0;
    gphi[n+0] =  -1.0;
    gphi[n+1] =  0.0;
    gphi[n+2] =  1.0;

    break;

  case 2:
    gphi[0] =  -3.0+4.0*x+4.0*y;
    gphi[2] =  -1.0+4.0*x;
    gphi[5] =  0.0;
    gphi[4] =  4.0*y;
    gphi[3] =  -4.0*y;
    gphi[1] =  4.0-8.0*x-4.0*y;
    gphi[n+0] =  -3.0+4.0*x+4.0*y;
    gphi[n+2] =  0.0;
    gphi[n+5] =  -1.0+4.0*y;
    gphi[n+4] =  4.0*x;
    gphi[n+3] =  4.0-4.0*x-8.0*y;
    gphi[n+1] =  -4.0*x;

    break;

  case 3:
    gphi[0] =  -11.0/2.0+18.0*x+18.0*y-27.0/2.0*x*x-27.0*x*y-27.0/2.0*y*y;
    gphi[3] =  1.0-9.0*x+27.0/2.0*x*x;
    gphi[9] =  0.0;
    gphi[6] =  -9.0/2.0*y+27.0*x*y;
    gphi[8] =  -9.0/2.0*y+27.0/2.0*y*y;
    gphi[7] =  9.0/2.0*y-27.0/2.0*y*y;
    gphi[4] =  -45.0/2.0*y+27.0*x*y+27.0*y*y;
    gphi[1] =  9.0-45.0*x-45.0/2.0*y+81.0/2.0*x*x+54.0*x*y+27.0/2.0*y*y;
    gphi[2] =  -9.0/2.0+36.0*x+9.0/2.0*y-81.0/2.0*x*x-27.0*x*y;
    gphi[5] =  27.0*y-54.0*x*y-27.0*y*y;
    gphi[n+0] =  -11.0/2.0+18.0*x+18.0*y-27.0/2.0*x*x-27.0*x*y-27.0/2.0*y*y;
    gphi[n+3] =  0.0;
    gphi[n+9] =  1.0-9.0*y+27.0/2.0*y*y;
    gphi[n+6] =  -9.0/2.0*x+27.0/2.0*x*x;
    gphi[n+8] =  -9.0/2.0*x+27.0*x*y;
    gphi[n+7] =  -9.0/2.0+9.0/2.0*x+36.0*y-27.0*x*y-81.0/2.0*y*y;
    gphi[n+4] =  9.0-45.0/2.0*x-45.0*y+27.0/2.0*x*x+54.0*x*y+81.0/2.0*y*y;
    gphi[n+1] =  -45.0/2.0*x+27.0*x*x+27.0*x*y;
    gphi[n+2] =  9.0/2.0*x-27.0/2.0*x*x;
    gphi[n+5] =  27.0*x-27.0*x*x-54.0*x*y;

    break;


    default:
    return cudaErrorNotSupported;
    break;
  }

  return cudaSuccess;
} 
