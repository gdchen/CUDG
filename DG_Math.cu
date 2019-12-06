/* This is the source file for the math lib 
 *
 * Author: Guodong Chen
 * Email:  cgderic@umich.edu
 * Last modified: 12/06/2019
 */ 

#include <stdlib.h>
#include "CUDA_Helper.cuh"
#include "DG_Math.cuh"
// Matrix multiplication 
// C = AxB
__device__ __host__ void  
DG_MxM_Set(int rA, int n, int cB, const double *A, const double *B, double *C)
{
  DG_cMxM_Set(1.0, rA, n, cB, A, B, C);
}


// C = cxAxB
__device__ __host__ void  
DG_cMxM_Set(double c, int rA, int n, int cB, const double *A, const double *B, double *C)
{
  int i, j, k;
  for (i=0; i<rA; i++){
    for (j=0; j<cB; j++){
      C[i*cB+j] = .0; 
      for (k=0; k<n; k++){
        C[i*cB+j] += c*A[i*n+k]*B[k*cB+j];
      }
    }
  }
}


// C += AxB
__device__ __host__ void
DG_MxM_Add(int rA, int n, int cB, const double *A, const double *B, double *C)
{
  DG_cMxM_Add(1.0, rA, n, cB, A, B, C); 
}


// C += cxAxB
__device__ __host__ void  
DG_cMxM_Add(double c, int rA, int n, int cB, const double *A, const double *B, double *C)
{
  int i, j, k;
  for (i=0; i<rA; i++){
    for (j=0; j<cB; j++){
      for (k=0; k<n; k++){
        C[i*cB+j] += c*A[i*n+k]*B[k*cB+j]; 
      }
    }
  }
}

// C -= AxB
__device__ __host__ void 
DG_MxM_Sub(int rA, int n, int cB, const double *A, const double *B, double *C)
{
  DG_cMxM_Add(-1.0, rA, n, cB, A, B, C); 
}


// C -= cxAxB
__device__ __host__ void 
DG_cMxM_Sub(double c, int rA, int n, int cB, const double *A, const double *B, double *C)
{
  DG_cMxM_Add(-c, rA, n, cB, A, B, C); 
}



// C = AxB^T
// the dimension if of op(A) and op(B)
// dim(A) = rAxn,  dim(B^T) = n*cB   <==>   dim(B) = cB*n 
__device__ __host__ void 
DG_MxMT_Set(int rA, int n, int cB, const double *A, const double *B, double *C)
{
  int i, j, k;    
  for (i=0; i<rA; i++){
    for (j=0; j<cB; j++){
      C[i*cB+j] = .0;    // initialization 
      for (k=0; k<n; k++){
        C[i*cB+j] += A[i*n+k]*B[j*n+k]; // B^T[k,j] == B[j,k]
      }
    }
  }
}


// C = A^TxB
// dim(A^T) = rAxn   <==>   dim(A) = n*rA
// dim(B) = n*cB
__device__ __host__ void 
DG_MTxM_Set(int rA, int n, int cB, double *A, double *B, double *C)
{
  int i, j, k; 
  for (i=0; i<rA; i++){
    for (j=0; j<cB; j++){
      C[i*cB+j] = .0;   // initialization 
      for (k=0; k<n; k++){
        C[i*cB+j] += A[k*rA+i]*B[k*cB+j];  // A^T[i,k] = A[k,i]
      }
    }
  }
}


// C += A^TxB
// dim(A^T) = rAxn   <==>   dim(A) = nxrA
// dim(B) = nxcB
__device__ __host__ void  
DG_MTxM_Add(int rA, int n, int cB, double *A, double *B, double *C)
{
  int i, j, k;
  for (i=0; i<rA; i++){
    for (j=0; j<cB; j++){
      for (k=0; k<n; k++){
        C[i*cB+j] += A[k*rA+i]*B[k*cB+j];  // A^T[i,k] = A[k,i]
      }
    }
  }
}

