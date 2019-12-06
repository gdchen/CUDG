/* This file is the header file for the math lib 
 * 
 * Author: Guodong 
 * Email: cgderic@umich.edu
 * Last modified: 12/06/2019
 */

#ifndef _DG_Math_
#define _DG_Math_ 

// Matrix multiplication 
// C = AxB
__device__ __host__ void  
DG_MxM_Set(int rA, int n, int cB, const double *A, const double *B, double *C); 

// C = cxAxB
__device__ __host__ void  
DG_cMxM_Set(double c, int rA, int n, int cB, const double *A, const double *B, double *C); 

// C += AxB
__device__ __host__ void 
DG_MxM_Add(int rA, int n, int cB, const double *A, const double *B, double *C); 

// C += cxAxB
__device__ __host__ void 
DG_cMxM_Add(double c, int rA, int n, int cB, const double *A, const double *B, double *C);

// C -= AxB
__device__ __host__ void  
DG_MxM_Sub(int rA, int n, int cB, const double *A, const double *B, double *C);

// C -= cxAxB
__device__ __host__ void  
DG_cMxM_Sub(double c, int rA, int n, int cB, const double *A, const double *B, double *C);

// C = AxB^T
__device__ __host__ void  
DG_MxMT_Set(int rA, int n, int cB, const double *A, const double *B, double *C);

// C = A^TxB
__device__ __host__ void  
DG_MTxM_Set(int rA, int n, int cB, double *A, double *B, double *C);

// C += A^TxB
__device__ __host__ void  
DG_MTxM_Add(int rA, int n, int cB, double *A, double *B, double *C);

// return matrix inverse to a new vector 
__device__ __host__ void 
DG_CreateInv(int n, const double *A, double *InvA);

// return matrix inverse into the original vector 
__device__ __host__ void 
DG_Inv(int n, double *A);




#endif
