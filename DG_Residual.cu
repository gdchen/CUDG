/* This is the source file for the DG_Residual, containing the function definitions 
 * foe the residual evaluations in DG
 * 
 * Author: Guodong Chen
 * Email: cgderic@umich.edu
 * Last modified: 12/07/2019
 */ 

#include <stdlib.h>
#include "DG_Mesh.cuh"
#include "DG_Quad.cuh"
#include "DG_Basis.cuh"
#include "DG_DataSet.cuh"
#include "DG_All.cuh"
#include "DG_Residual.cuh"
#include "CUDA_Helper.cuh"


// flux function 
__device__ 
int calculateFlux(double *Fx, double *Fy, int nq2, const double *Uxy)
{
  // All inputs should be allocated before passed in
  // Fx~Fy = [nq2*NUM_OF_STATES], Uxy[nq2*NUM_OF_STATES]
  int i;
  double q1, q2, q3, q4, p;
  for (i=0; i<nq2; i++){
    q1 = Uxy[i*NUM_OF_STATES+0]; 
    q2 = Uxy[i*NUM_OF_STATES+1];
    q3 = Uxy[i*NUM_OF_STATES+2];
    q4 = Uxy[i*NUM_OF_STATES+3];
    p = (GAMMA-1)*(q4-0.5*(q2*q2+q3*q3)/q1); 
    Fx[i*NUM_OF_STATES+0] = q2;
    Fx[i*NUM_OF_STATES+1] = q2*q2/q1+p;
    Fx[i*NUM_OF_STATES+2] = q2*q3/q1;
    Fx[i*NUM_OF_STATES+3] = q2/q1*(q4+p);

    Fy[i*NUM_OF_STATES+0] = q3; 
    Fy[i*NUM_OF_STATES+1] = q2*q3/q1;
    Fy[i*NUM_OF_STATES+2] = q3*q3/q1+p;
    Fy[i*NUM_OF_STATES+3] = q3/q1*(q4+p);

  }  
  return 0;

};



// roe flux 
__device__ 
int getFhat(double *Fhat, const double *UL, const double *UR, const double *normal)
{
  double rL = UL[0]; //printf("%f \n", rL);
  double uL = UL[1]/rL; 
  double vL = UL[2]/rL;
  double unL = uL*normal[0] + vL*normal[1];      //printf("unL: %f \n", unL);
  double qL = sqrt(UL[1]*UL[1]+UL[2]*UL[2])/rL;  //printf("qL: %f \n", qL*rL);
  double pL = (GAMMA-1)*(UL[3]-0.5*rL*qL*qL);    //printf("pL: %f \n", pL);
  if ((pL<=0)||(rL<=0)) {printf("Non-physical 1 state!\n"); return -1;}
  double rHL = UL[3]+pL;
  double HL = rHL/rL;
  //double cL = sqrt(GAMMA*pL/rL);

  //Left flu 
  double FL[NUM_OF_STATES];
  FL[0] = rL*unL;
  FL[1] = UL[1]*unL + pL*normal[0];
  FL[2] = UL[2]*unL + pL*normal[1];
  FL[3] = rHL*unL;

  // process right state
  double rR = UR[0];
  double uR = UR[1]/rR;
  double vR = UR[2]/rR;
  double unR = uR*normal[0] + vR*normal[1];
  double qR = sqrt(UR[1]*UR[1]+UR[2]*UR[2])/rR;
  double pR = (GAMMA-1)*(UR[3]-0.5*rR*qR*qR);
  if ((pR<=0)||(rR<=0)) {printf("Non-physical 2 state! \n"); return -1;}
  double rHR = UR[3] + pR;
  double HR = rHR/rR;
  //double cR = sqrt(GAMMA*pR/rR);

  //right flux 
  double FR[NUM_OF_STATES];
  FR[0] = rR*unR;
  FR[1] = UR[1]*unR + pR*normal[0];
  FR[2] = UR[2]*unR + pR*normal[1];
  FR[3] = rHR*unR;

  // diff in states 
  double du[NUM_OF_STATES];
  int i; 
  for (i=0; i<NUM_OF_STATES; i++) du[i] = UR[i] - UL[i];
  // Roe average 
  double di = sqrt(rR/rL);
  double d1 = 1.0/(1.0+di);
  double ui = (di*uR + uL)*d1;
  double vi = (di*vR + vL)*d1;
  double Hi = (di*HR + HL)*d1; 

  double af = 0.5*(ui*ui+vi*vi);
  double ucp = ui*normal[0] + vi*normal[1];
  double c2 = (GAMMA-1)*(Hi-af);
  if (c2<=0) {printf("Non-physical 3 state! \n"); return -1;}
  double ci = sqrt(c2);
  double ci1 = 1.0/ci;

  double l[3];
  l[0] = ucp + ci;
  l[1] = ucp - ci;
  l[2] = ucp;

  // entropy fix 
  double epsilon = ci*.1;
  for (i=0; i<3; i++){
    if ((l[i]<epsilon) && (l[i]>-epsilon))
      l[i] = 0.5*(epsilon+l[i]*l[i]/epsilon);
    l[i] = fabs(l[i]);
  } 
  double l3 = l[2];

  // average and half-difference of 1st and 2nd eigs 
  double s1 = 0.5*(l[0]+l[1]);
  double s2 = 0.5*(l[0]-l[1]);

  // left eigenvector product generators 
  double G1 = (GAMMA-1)*(af*du[0]-ui*du[1]-vi*du[2] + du[3]);
  double G2 = -ucp*du[0] + du[1]*normal[0] + du[2]*normal[1];

  // requireed functions of G1 and G2
  double C1 = G1*(s1 - l3)*ci1*ci1 + G2*s2*ci1;
  double C2 = G1*s2*ci1            + G2*(s1 - l3);

  Fhat[0] = 0.5*(FL[0]+FR[0]) - 0.5*(l3*du[0] + C1);
  Fhat[1] = 0.5*(FL[1]+FR[1]) - 0.5*(l3*du[1] + C1*ui + C2*normal[0]);
  Fhat[2] = 0.5*(FL[2]+FR[2]) - 0.5*(l3*du[2] + C1*vi + C2*normal[1]);
  Fhat[3] = 0.5*(FL[3]+FR[3]) - 0.5*(l3*du[3] + C1*Hi + C2*ucp);


  double smag = 0;
  for (i=0; i<3; i++) smag = (smag > l[i]) ? smag:l[i]; 
  //printf("smag: %f \n", smag);

  return 0;





}


// calculate roe flux for each edge 
__device__ 
int calculateFhat(double *Fhat, int nq1, const double *UL, const double *UR, const double *normal)
{
  int i;
  for (i=0; i<nq1; i++)
    getFhat(Fhat+i*NUM_OF_STATES, UL+i*NUM_OF_STATES, UR+i*NUM_OF_STATES, normal);
  return 0; 
};




/* kernel to calculate the volume residual */
__global__ void 
calculateVolumeRes(const DG_All *All, double **State, double **R){
  
  int tid = threadIdx.x;   // thread index
  int gid = blockIdx.x*blockDim.x + tid;   // global index
  
  DG_BasisData *BasisData = All->BasisData;   // TODO: make it in constant memory 
  // shared memory 
  int np = BasisData->np; 
  int nq2 = BasisData->nq2; 
  int nElem = All->Mesh->nElem; 
  // 
  double InvJac[4];  // Inverse Jacobian, only read in once, cant be shared memory 
  int i, j; 
  // global(physical) gradicents, these two array are thread locally dynamic memory 
  double *GPhix = (double *) malloc(nq2*np*sizeof(double)); 
  double *GPhiy = (double *) malloc(nq2*np*sizeof(double)); 
  // Diagonal matrix TODO: make it in shared memory 
  double *Dwq = (double *) malloc(nq2*nq2*sizeof(double)); 
  for (i=0; i<nq2; i++){
    for (j=0; j<nq2; j++)
      Dwq[i*nq2+j] = 0;
    Dwq[i*nq2+i] = BasisData->wq2[i]; 
  }
  // get states at interior quad points, thread local dynamic memory  
  double *Uxy = (double *)malloc(nq2*NUM_OF_STATES*sizeof(double)); 
  double *Fx  = (double *)malloc(nq2*NUM_OF_STATES*sizeof(double));
  double *Fy  = (double *)malloc(nq2*NUM_OF_STATES*sizeof(double));
  // temp matrix for flux calculation 
  double *temp = (double *)malloc(np*nq2*sizeof(double)); 
  // all the local dynamically allocated memory do not need initialization
  // as they were set in the function called later
  

  /* each thread calculates the interior volume flux */
  if (gid < nElem){
    getIntQuadStates(Uxy, State[gid], BasisData); 
    calculateFlux(Fx, Fy, nq2, Uxy);
    // read in once 
    for (i=0; i<4; i++)
      InvJac[i] = All->Mesh->InvJac[gid][i]; 

    for (i=0; i<nq2; i++){
      for (j=0; j<np; j++){
        GPhix[i*np+j] = BasisData->GPhix[i*np+j]*InvJac[0] + BasisData->GPhiy[i*np+j]*InvJac[2];
        GPhiy[i*np+j] = BasisData->GPhix[i*np+j]*InvJac[1] + BasisData->GPhiy[i*np+j]*InvJac[3]; 
      }
    }
    //
    DG_MTxM_Set(np, nq2, nq2, GPhix, Dwq, temp); 
    DG_MxM_Set (np, nq2, NUM_OF_STATES, temp, Fx, R[gid]);
    DG_MTxM_Set(np, nq2, nq2, GPhiy, Dwq, temp);
    DG_MxM_Add (np, nq2, NUM_OF_STATES, temp, Fy, R[gid]); 
  }

  free(GPhix);  free(GPhiy);  free(Uxy);  free(Fx);  free(Fy); free(temp);
  free(Dwq);  // TODO: move this into shared memory or constant memory 



}





/* kernel to calculate the face residual */
__global__ void  
calculateFaceRes(const DG_All *All, double **State, double **RfL, double **RfR){
  
  int tid = threadIdx.x; 
  int gid = blockIdx.x*blockDim.x + tid; 
  int i, j; 
  DG_BasisData *BasisData = All->BasisData; 
  int np  = BasisData->np; 
  int nq1 = BasisData->nq1; 
  double *UL = (double *)malloc(nq1*NUM_OF_STATES*sizeof(double)); 
  double *UR = (double *)malloc(nq1*NUM_OF_STATES*sizeof(double)); 
  // Diagonal matrix TODO: make this shared or constant memory 
  double *Dwq1 = (double *)malloc(nq1*nq1*sizeof(double)); 
  for (i=0; i<nq1; i++){
    for (j=0; j<nq1; j++)
      Dwq1[i*nq1+j] = 0;
    Dwq1[i*nq1+i] = BasisData->wq1[i]; 
  }
  double *Fhat = (double *)malloc(nq1*NUM_OF_STATES*sizeof(double));
  double *temp = (double *)malloc(np*nq1*sizeof(double)); 
 
  int ElemL, ElemR, edge; 
  if (gid < All->Mesh->nIFace){
    ElemL = All->Mesh->IFace[gid].ElemL; 
    ElemR = All->Mesh->IFace[gid].ElemR; 
    edge  = All->Mesh->IFace[gid].EdgeL;   // edgeL = edgeR
    getEdgeQuadStates(UL, UR, edge, State[ElemL], State[ElemR], BasisData); 
    calculateFhat(Fhat, nq1, UL, UR, All->Mesh->normal+2*gid);
    
    DG_MTxM_Set(np, nq1, nq1, BasisData->EdgePhiL[edge], Dwq1, temp);
    DG_cMxM_Set(All->Mesh->Length[gid], np, nq1, NUM_OF_STATES, temp, Fhat, RfL[gid]); // sub later

    DG_MTxM_Set(np, nq1, nq1, BasisData->EdgePhiR[edge], Dwq1, temp); 
    DG_cMxM_Set(All->Mesh->Length[gid], np, nq1, NUM_OF_STATES, temp, Fhat, RfR[gid]); // ad later
  }

  free(UL);  free(UR);  free(Fhat);  free(temp); 
  free(Dwq1);  // TODO: make it in constant memory or shared memory 

}



/* kernel to add volume and face residuals, and convert to RHS for time integration */
__global__ void 
addRes(const DG_All *All, double **R, double **RfL, double **RfR){

  int tid = threadIdx.x; 
  int gid = blockIdx.x * blockDim.x + tid; 
  int i, edge;
  int ndof = All->BasisData->np * NUM_OF_STATES; 
  //int E2F[3]; 
  //for (edge=0; edge<3; edge++)
  //  E2F[edge] = All->Mesh->E2F[gid]i[edge]; 
  int ElemL, ElemR; 
  int gface; 

  if (gid < All->Mesh->nElem){
    for (edge=0; edge<3; edge++){
      gface = All->Mesh->E2F[gid][edge]; 
      ElemL = All->Mesh->IFace[gface].ElemL; 
      ElemR = All->Mesh->IFace[gface].ElemR; 
      //ElemL = All->Mesh->IFace[All->Mesh->E2F[gid][edge]].ElemL; 
      //ElemR = All->Mesh->IFace[All->Mesh->E2F[gid][edge]].ElemR;
      if (gid == ElemL)
        for (i=0; i<ndof; i++)
          R[gid][i] -= RfL[gface][i];  
      if (gid == ElemR)
        for (i=0; i<ndof; i++)
          R[gid][i] += RfR[gface][i];
    }
  }

}


__global__ void 
getResAllAtOnce(const DG_All *All, double **State, double **R){

}

__global__ void 
Res2RHS(const DG_All *All, double **R, double **f){

  int tid = threadIdx.x; 
  int gid = blockIdx.x * blockDim.x + tid; 
  int np = All->BasisData->np; 
  if (gid < All->Mesh->nElem){
    DG_MxM_Set(np, np, NUM_OF_STATES, All->DataSet->InvMassMatrix[gid], R[gid], f[gid]); 
  }

}


/* kernel to update intermediate states in RK4 */
__global__ void 
rk4_inter(DG_All *All, double **State, double dt, double **f){
  int tid = threadIdx.x; 
  int gid = blockIdx.x * blockDim.x + tid; 
  int ndof = All->BasisData->np * NUM_OF_STATES; 
  int i; 
  if (gid < All->Mesh->nElem){
    for (i=0; i<ndof; i++)
      State[gid][i] = All->DataSet->State[gid][i] + dt*f[gid][i]; 
  }

}


/* kernel to update the final states in rk4 */
__global__ void
rk4_final(DG_All *All, double dt, double **f0, double **f1, double **f2, double **f3)
{

  int tid = threadIdx.x; 
  int gid = blockIdx.x * blockDim.x + tid; 
  int ndof = All->BasisData->np * NUM_OF_STATES; 
  int i; 
  if (gid < All->Mesh->nElem){
    for (i=0; i<ndof; i++)
      All->DataSet->State[gid][i] += dt*(f0[gid][i] + 2*f1[gid][i] + 2*f2[gid][i] + f3[gid][i]);

  }

}


/* host function to lunch kernels performing RK4 time integration */
cudaError_t
DG_RK4(DG_All *All, double dt, int Nt){
  
  int nElem = All->Mesh->nElem;  // # of elem 
  int nIFace = All->Mesh->nIFace; // # of faces 
  int np = All->BasisData->np;   // dof per elem 
  int i, j; 
  // temp states, residual, rhs for rk4
  double **U, **R, **f0, **f1, **f2, **f3; 
  double *tempU, *tempR, *tempf0, *tempf1, *tempf2, *tempf3; 
  // memory allocation 
  CUDA_CALL(cudaMallocManaged(&U,  nElem*sizeof(double *)));
  CUDA_CALL(cudaMallocManaged(&R,  nElem*sizeof(double *)));
  CUDA_CALL(cudaMallocManaged(&f0, nElem*sizeof(double *)));
  CUDA_CALL(cudaMallocManaged(&f1, nElem*sizeof(double *)));
  CUDA_CALL(cudaMallocManaged(&f2, nElem*sizeof(double *)));
  CUDA_CALL(cudaMallocManaged(&f3, nElem*sizeof(double *)));

  CUDA_CALL(cudaMallocManaged(&tempU,  nElem*np*NUM_OF_STATES*sizeof(double)));
  CUDA_CALL(cudaMallocManaged(&tempR,  nElem*np*NUM_OF_STATES*sizeof(double)));
  CUDA_CALL(cudaMallocManaged(&tempf0, nElem*np*NUM_OF_STATES*sizeof(double))); 
  CUDA_CALL(cudaMallocManaged(&tempf1, nElem*np*NUM_OF_STATES*sizeof(double))); 
  CUDA_CALL(cudaMallocManaged(&tempf2, nElem*np*NUM_OF_STATES*sizeof(double))); 
  CUDA_CALL(cudaMallocManaged(&tempf3, nElem*np*NUM_OF_STATES*sizeof(double))); 

  
  double **RfL, **RfR; 
  double *tempRfL, *tempRfR; 
  CUDA_CALL(cudaMallocManaged(&RfL, nIFace*sizeof(double *)));
  CUDA_CALL(cudaMallocManaged(&RfR, nIFace*sizeof(double *)));
  
  CUDA_CALL(cudaMallocManaged(&tempRfL, nIFace*np*NUM_OF_STATES*sizeof(double)));
  CUDA_CALL(cudaMallocManaged(&tempRfR, nIFace*np*NUM_OF_STATES*sizeof(double)));


  // allocation and initialization 
  for (i=0; i<nElem; i++){
    U[i]  = tempU  + i*np*NUM_OF_STATES; 
    R[i]  = tempR  + i*np*NUM_OF_STATES; 
    f0[i] = tempf0 + i*np*NUM_OF_STATES; 
    f1[i] = tempf1 + i*np*NUM_OF_STATES; 
    f2[i] = tempf2 + i*np*NUM_OF_STATES; 
    f3[i] = tempf3 + i*np*NUM_OF_STATES;  
    for (j=0; j<np*NUM_OF_STATES; j++){
      U[i][j] = 0;
      R[i][j] = 0; 
      f0[i][j] = 0;  f1[i][j] = 0; f2[i][j] = 0; f3[i][j] = 0;
    }
  }

  for (i=0; i<nIFace; i++){
    RfL[i] = tempRfL + i*np*NUM_OF_STATES; 
    RfR[i] = tempRfR + i*np*NUM_OF_STATES;
  }


  int threadPerBlock = 512;
  int elemBlock = (nElem + threadPerBlock - 1)/threadPerBlock; 
  int faceBlock = (nIFace + threadPerBlock -1)/threadPerBlock; 
  

  printf("elem kernel lunch (%d,%d)\n",elemBlock, threadPerBlock);  
  printf("face kernel lunch (%d,%d)\n",faceBlock, threadPerBlock);  
  // async kernel luncah 
  
  int t = 0; 
  for (t=0; t<Nt; t++){
    // first we need to copy the states data 
    CUDA_CALL(cudaMemcpy(U[0], All->DataSet->State[0], nElem*np*NUM_OF_STATES*sizeof(double), 
              cudaMemcpyDeviceToDevice)); 
    calculateVolumeRes <<<elemBlock, threadPerBlock>>> (All, U, R);
    CUDA_CALL(cudaGetLastError());
    calculateFaceRes   <<<faceBlock, threadPerBlock>>> (All, U, RfL, RfR); 
    addRes             <<<elemBlock, threadPerBlock>>> (All, R, RfL, RfR); 
    Res2RHS            <<<elemBlock, threadPerBlock>>> (All, R, f0); 
    rk4_inter          <<<elemBlock, threadPerBlock>>> (All, U, dt/2, f0);

    calculateVolumeRes <<<elemBlock, threadPerBlock>>> (All, U, R);
    calculateFaceRes   <<<faceBlock, threadPerBlock>>> (All, U, RfL, RfR); 
    addRes             <<<elemBlock, threadPerBlock>>> (All, R, RfL, RfR); 
    Res2RHS            <<<elemBlock, threadPerBlock>>> (All, R, f1); 
    rk4_inter          <<<elemBlock, threadPerBlock>>> (All, U, dt/2, f1);

    calculateVolumeRes <<<elemBlock, threadPerBlock>>> (All, U, R);
    calculateFaceRes   <<<faceBlock, threadPerBlock>>> (All, U, RfL, RfR); 
    addRes             <<<elemBlock, threadPerBlock>>> (All, R, RfL, RfR); 
    Res2RHS            <<<elemBlock, threadPerBlock>>> (All, R, f2); 
    rk4_inter          <<<elemBlock, threadPerBlock>>> (All, U, dt, f2);

    calculateVolumeRes <<<elemBlock, threadPerBlock>>> (All, U, R);
    calculateFaceRes   <<<faceBlock, threadPerBlock>>> (All, U, RfL, RfR); 
    addRes             <<<elemBlock, threadPerBlock>>> (All, R, RfL, RfR); 
    Res2RHS            <<<elemBlock, threadPerBlock>>> (All, R, f3); 
    rk4_final          <<<elemBlock, threadPerBlock>>> (All, dt/6, f0, f1, f2, f3);
  }

  // free memory 
  CUDA_CALL(cudaFree(tempU));  CUDA_CALL(cudaFree(U));
  CUDA_CALL(cudaFree(tempR));  CUDA_CALL(cudaFree(R));
  CUDA_CALL(cudaFree(tempf0)); CUDA_CALL(cudaFree(f0));
  CUDA_CALL(cudaFree(tempf1)); CUDA_CALL(cudaFree(f1));
  CUDA_CALL(cudaFree(tempf2)); CUDA_CALL(cudaFree(f2));
  CUDA_CALL(cudaFree(tempf3)); CUDA_CALL(cudaFree(f3));
  
  CUDA_CALL(cudaFree(tempRfL)); CUDA_CALL(cudaFree(RfL));
  CUDA_CALL(cudaFree(tempRfR)); CUDA_CALL(cudaFree(RfR));

  return cudaSuccess;

}
