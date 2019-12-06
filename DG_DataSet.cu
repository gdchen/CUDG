/* This file contains the implementation of the DG_DataSet methods 
 *
 * Author: Guodong Chen
 * Email: cgderic@umich.edu 
 * Last modified: 12/05/2019
 */ 

#include "stdlib.h"
#include "DG_Const.cuh"
#include "DG_Mesh.cuh"
#include "DG_Quad.cuh"
#include "DG_Basis.cuh"
#include "DG_DataSet.cuh"



cudaError_t initDataSet(DG_DataSet *DataSet)
{
  DataSet->nElem = 0;
  DataSet->order = 0;
  DataSet->State = NULL;
  DataSet->MassMatrix = NULL;
  DataSet->InvMassMatrix = NULL; 
  return cudaSuccess; 
}





cudaError_t createDataSet(DG_DataSet **pDataSet)
{
  CUDA_CALL(cudaMallocManaged(pDataSet, sizeof(DG_DataSet))); 
  CUDA_CALL(initDataSet(*pDataSet));
  return cudaSuccess; 
}



cudaError_t freeDataSet(DG_DataSet *DataSet)
{
  int nElem = DataSet->nElem;
  int i;
  if (DataSet->MassMatrix != NULL){
    for (i=0; i<nElem; i++){
      CUDA_CALL(cudaFree(DataSet->MassMatrix[i])); 
      CUDA_CALL(cudaFree(DataSet->InvMassMatrix[i]));
    }
  }
  if (DataSet->State != NULL){
    for (i=0; i<nElem; i++) 
      CUDA_CALL(cudaFree(DataSet->State[i]));
  }
  free(DataSet->State);  
  free(DataSet->MassMatrix);  
  free(DataSet->InvMassMatrix);
  free(DataSet);
  return cudaSuccess; 
}



cudaError_t
computeMassMatrix(DG_DataSet *DataSet, const DG_Mesh *Mesh, const DG_BasisData *BasisData)
{
  int nElem = Mesh->nElem;
  int order = BasisData->order;
  DataSet->nElem = nElem;
  DataSet->order = order; 
  int np = BasisData->np;
  int nq2 = BasisData->nq2; 
  double *Phi = BasisData->Phi; 
  double *wq2 = BasisData->wq2;  
  // allocate the memory for the massmatrix and inverse mass matrix 
  CUDA_CALL(cudaMallocManaged(&(DataSet->MassMatrix), nElem*sizeof(double *)));
  CUDA_CALL(cudaMallocManaged(&(DataSet->InvMassMatrix), nElem*sizeof(double *))); 

  int n, i, j, q2;  
  for (n=0; n<nElem; n++){
    CUDA_CALL(cudaMallocManaged(&(DataSet->MassMatrix[n]), np*np*sizeof(double)));
    CUDA_CALL(cudaMallocManaged(&(DataSet->InvMassMatrix[n], np*np*sizeof(double)))); 
    for (i=0; i<np; i++){
      for (j=0; j<np; j++){
        DataSet->MassMatrix[n][i*np+j] = 0.0;    // Initialization  
        for (q2=0; q2<nq2; q2++){
          DataSet->MassMatrix[n][i*np+j] += Phi[q2*np+i]*Phi[q2*np+j]*wq2[q2]; 
        }
        DataSet->MassMatrix[n][i*np+j] *= Mesh->detJ[n];
        DataSet->InvMassMatrix[n][i*np+j] = DataSet->MassMatrix[n][i*np+j];

      }
    }
    // invert the massmatrix to get InvMassMatrix 
    DG_Inv(np, DataSet->InvMassMatrix[n]); 
  }

}


/* initialize the DataSet using interpolation */
cudaError_t interpolateIC(DG_DataSet *DataSet, const DG_Mesh *Mesh)
{
  int nElem = DataSet->nElem;
  int order = DataSet->order; 
  int np = (order+1)*(order+2)/2; 
  int i, j;
  // no need to call cudaMalloc as the memory pointer passed in 
  // is already on the device and free by the device directly 
  double **xyGlobal = (double **)malloc(nElem*np*sizeof(double *));
  for (i=0; i<nElem*np; i++) xyGlobal[i] = (double *)malloc(2*sizeof(double));
  getGlobalLagrangeNodes(order, Mesh, xyGlobal);
  // this chunck of memory is accessiable by both CPU and GPU 
  // use cudaMalloc instead 
  CUDA_CALL(cudaMallocManaged(&(DataSet->State), nElem*sizeof(double *))); 

  double f0, f1, f2, rho, u, v, p;
  for (i=0; i<nElem; i++){
    CUDA_CALL(cudaMallocManaged(&(DataSet->State[i]), np*NUM_OF_STATES*sizeof(double))); 
    for (j=0; j<np; j++){
      f0 = getf0(xyGlobal[i*np+j]);
      f1 = getf1(f0);
      f2 = getf2(f0);
      rho = RHO_INF*pow(f1, 1.0/(GAMMA-1));
      u = U_INF - f2*(xyGlobal[i*np+j][1]-X_ORIGINAL[1]);
      v = V_INF + f2*(xyGlobal[i*np+j][0]-X_ORIGINAL[0]);
      p = P_INF*pow(f1, GAMMA/(GAMMA-1));
      DataSet->State[i][j*NUM_OF_STATES+0] = rho;
      DataSet->State[i][j*NUM_OF_STATES+1] = rho*u;
      DataSet->State[i][j*NUM_OF_STATES+2] = rho*v;
      DataSet->State[i][j*NUM_OF_STATES+3] = p/(GAMMA-1) + 0.5*rho*(u*u+v*v);
    }
  }
  for (i=0; i<nElem*np; i++) free(xyGlobal[i]); free(xyGlobal); 
  return cudaSuccess; 

}


cudaError_t 
getIntQuadStates(double *Uxy, const double *State, const DG_BasisData *BasisData)
{
  // All of the inputs should be allocated before passed in 
  // Uxy[nq2*NUM_OF_STATES]
  int np = BasisData->np; 
  int nq2 = BasisData->nq2;
  double *Phi = BasisData->Phi;
  DG_MxM_Set(nq2, np, NUM_OF_STATES, Phi, State, Uxy);
  return cudaSuccess;
}

// get states at edge quad points 
cudaError_t 
getEdgeQuadStates(double *UL, double *UR, int edge, const double *StateL, const double *StateR,
                  const DG_BasisData *BasisData)
{
  int np = BasisData->np;
  int nq1 = BasisData->nq1;
  double **EdgePhiL = BasisData->EdgePhiL;
  double **EdgePhiR = BasisData->EdgePhiR; 
  DG_MxM_Set(nq1, np, NUM_OF_STATES, EdgePhiL[edge], StateL, UL);
  DG_MxM_Set(nq1, np, NUM_OF_STATES, EdgePhiR[edge], StateR, UR); 
  return cudaSuccess;
}






// Lagrange Nodes 
cudaError_t getLagrangeNodes(int order, double **xy)
{
  // xy should be allocated before passed in, xy[np][2]
  int i,j, counter; 
  // no need to cudaMalloc as it's local memory 
  // can be alloc and free by either CPU or GPU in the same function 
  double *x = (double *)malloc((order+1)*sizeof(double)); // 1d Lagrange nodes 
  if (order == 0) x[0] = 0.333333333333333333;
  else {for (i=0; i<order+1; i++) x[i] = 1.0/order*i;} 
  counter = 0;
  for (i=0; i<order+1; i++){
    for (j=0; j<order+1-i; j++){
      xy[counter][0] = x[j];
      xy[counter][1] = x[i];
      counter++;
    }
  }
  free(x); 
  return cudaSuccess;
}

// global Lagrange nodes 
cudaError_t 
getGlobalLagrangeNodes(int order, const DG_Mesh *Mesh, double **xyGlobal)
{ 
  int np = (order+1)*(order+2)/2;
  int nElem = Mesh->nElem;
  double **coord = Mesh->coord;
  int **E2N = Mesh->E2N;
  double **Jac = Mesh->Jac; 
  int i, j;
  double **xy;
  xy = (double **)malloc(np*sizeof(double *));
  for (i=0; i<np; i++) xy[i] = (double *)malloc(2*sizeof(double));
  double *x0; 
  getLagrangeNodes(order, xy);  
  for (i=0; i<nElem; i++){
    x0 = coord[E2N[i][0]]; 
    for (j=0; j<np; j++){
      xyGlobal[i*np+j][0] = x0[0] + xy[j][0]*Jac[i][0] + xy[j][1]*Jac[i][1];
      xyGlobal[i*np+j][1] = x0[1] + xy[j][0]*Jac[i][2] + xy[j][1]*Jac[i][3];
    }
  }

  for (i=0; i<np; i++) free(xy[i]); free(xy);
  return cudaSuccess;
}


// global quad points 
int getGlobalQuadPoints(double **xyGlobal, const DG_Mesh *Mesh, const DG_BasisData *BasisData)
{
  int nq2 = BasisData->nq2; 
  double *xyq = BasisData->xyq; 
  int nElem = Mesh->nElem;
  double **coord = Mesh->coord;
  int **E2N = Mesh->E2N;
  double **Jac = Mesh->Jac;
  int i,j;
  double *x0, *xy;  
  for (i=0; i<nElem; i++)
  {
    x0 = coord[E2N[i][0]]; 
    for (j=0; j<nq2; j++){
      xy = xyq+2*j; 
      xyGlobal[i*nq2+j][0] = x0[0] + xy[0]*Jac[i][0] + xy[1]*Jac[i][1]; 
      xyGlobal[i*nq2+j][1] = x0[1] + xy[0]*Jac[i][2] + xy[1]*Jac[i][3];  
    }
  }
  return cudaSuccess; 
}





/*Functions of exact solution*/
double getf0(double *x)
{
  double abs =   (x[0]-X_ORIGINAL[0])*(x[0]-X_ORIGINAL[0]) 
               + (x[1]-X_ORIGINAL[1])*(x[1]-X_ORIGINAL[1]);
  double f0 = 1.0 - abs/RC/RC; 
  return f0;
}
double getf1(double f0)
{
  double f1;
  f1 = 1.0 -EPSILON*EPSILON*(GAMMA-1.0)*M_INF*M_INF*exp(f0)/8.0/PI/PI;
  return f1; 
}
double getf2(double f0)
{
  double f2; 
  f2 = EPSILON*U_ABS*exp(f0/2.0)/2.0/PI/RC;
  return f2;

}
