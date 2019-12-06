#ifndef _DG_Math_
#define _DG_Math_ 

#include <cblas.h>
#include <lapacke.h>
// Math library for DG_solver

// Matrix multiplication 
// C = AxB
void DG_MxM_Set(int rA, int n, int cB, const double *A, const double *B, double *C)
{
  double alpha = 1.0;  
  double beta = 0.0;
  int lda = n;
  int ldb = cB; 
  int ldc = cB; 
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                 rA, cB, n, alpha, A,
                 lda, B, ldb,
                 beta, C, ldc);

}


// C = cxAxB
void DG_cMxM_Set(double c, int rA, int n, int cB, const double *A, const double *B, double *C)
{
  double alpha = c;  
  double beta = 0.0;
  int lda = n;
  int ldb = cB; 
  int ldc = cB; 
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                 rA, cB, n, alpha, A,
                 lda, B, ldb,
                 beta, C, ldc);

}


// C += AxB
void DG_MxM_Add(int rA, int n, int cB, const double *A, const double *B, double *C)
{
  double alpha = 1.0; 
  double beta = 1.0; 
  int lda = n;
  int ldb = cB; 
  int ldc = cB; 
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                 rA, cB, n, alpha, A,
                 lda, B, ldb,
                 beta, C, ldc);

}


// C += cxAxB
void DG_cMxM_Add(double c, int rA, int n, int cB, const double *A, const double *B, double *C)
{
  double alpha = c; 
  double beta = 1.0; 
  int lda = n;
  int ldb = cB; 
  int ldc = cB; 
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                 rA, cB, n, alpha, A,
                 lda, B, ldb,
                 beta, C, ldc);

}




// C = -AxB
void DG_MxM_Sub(int rA, int n, int cB, const double *A, const double *B, double *C)
{
  double alpha = -1.0;
  double beta = 1.0;
  int lda = n;
  int ldb = cB; 
  int ldc = cB; 
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                 rA, cB, n, alpha, A,
                 lda, B, ldb,
                 beta, C, ldc);
}


// C -= cxAxB
void DG_cMxM_Sub(double c, int rA, int n, int cB, const double *A, const double *B, double *C)
{
  double alpha = -c;
  double beta = 1.0;
  int lda = n;
  int ldb = cB; 
  int ldc = cB; 
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                 rA, cB, n, alpha, A,
                 lda, B, ldb,
                 beta, C, ldc);
}



// C = AxB^T
void DG_MxMT_Set(int rA, int n, int cB, const double *A, const double *B, double *C)
{
  double alpha = 1.0;
  double beta = 0.0; 
  int lda = n;
  int ldb = n;
  int ldc = cB;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                 rA, cB, n, alpha, A,
                 lda, B, ldb,
                 beta, C, ldc);


}


// C = A^TxB
void DG_MTxM_Set(int rA, int n, int cB, double *A, double *B, double *C)
{
  double alpha = 1.0;
  double beta = 0.0; 
  int lda = rA;
  int ldb = cB;
  int ldc = cB;
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                 rA, cB, n, alpha, A,
                 lda, B, ldb,
                 beta, C, ldc);

}


// C += A^TxB
void DG_MTxM_Add(int rA, int n, int cB, double *A, double *B, double *C)
{
  double alpha = 1.0;
  double beta = 1.0; 
  int lda = rA;
  int ldb = cB;
  int ldc = cB;
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                 rA, cB, n, alpha, A,
                 lda, B, ldb,
                 beta, C, ldc);

}




// return matrix inverse to a new vector 
int DG_CreateInv(int n, const double *A, double *InvA)
{
  // all the inputs should be allocated 
  int info, i, j; 
  int ipiv[n];
  for (i=0; i<n; i++){
    for (j=0; j<n; j++) InvA[i*n+j] = A[i*n+j];
  }
  if(!(info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, InvA, n, ipiv)))
    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, InvA, n, ipiv);
  if (info!= 0) printf("Error Inverse\n");
  return info;

/*  lapack_int LAPACKE_dgetri( int matrix_layout, lapack_int n, double* a,
                           lapack_int lda, const lapack_int* ipiv );

  lapack_int LAPACKE_dgetrf( int matrix_layout, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, lapack_int* ipiv );*/
}


// return matrix inverse into the original vector 
int DG_Inv(int n, double *A)
{
  int info;
  int ipiv[n]; 
  if (!(info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A, n, ipiv)))
    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, A, n, ipiv);
  if (info != 0) printf("Error Inverse !\n");
  return info;
}




#endif