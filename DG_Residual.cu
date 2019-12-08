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
calculateVolumeRes(const DG_All *All, const double **State, double **R){
  
}





/* kernel to calculate the face residual */
__global__ void  
calculateFaceRes(const DG_All *All, const double **State, double **Rf){


}



/* kernel to add volume and face residuals, and convert to RHS for time integration */
__global__ void 
addRes2f(const DG_All *All, double **R, double **Rf, double **f){

}

/* kernel to update intermediate states in RK4 */
__global__ void 
rk4_inter(DG_All *All, double **State, double dt, double **f){

}


/* kernel to update the final states in rk4 */
__global__ void
rk4_final(DG_All *All, double dt, double **f0, double **f1, double **f2, double **f3)
{

}


/* host function to lunch kernels performing RK4 time integration */
__host__ cudaError_t
DG_RK4(DG_All *All){
  
  int nElem = All->Mesh->nElem;  // # of elem 
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
  // allocation and initialization 
  for (i=0; i<nElem; i++){
    U[i] = tempU   + i*np*NUM_OF_STATES; 
    R[i] = tempR   + i*np*NUM_OF_STATES; 
    f0[i] = tempf0 + i*np*NUM_OF_STATES; 
    f1[i] = tempf1 + i*np*NUM_OF_STATES; 
    f2[i] = tempf2 + i*np*NUM_OF_STATES; 
    f3[i] = tempf3 + i*np*NUM_OF_STATES;  
  }


  // async kernel luncah 
//  calculateVolumeRes <<<1,1>>>(); 
//  calculateFaceRes <<<1,1>>>(); 
//  // sync kernel lunch 
//  addRes2f <<<1,1>>> (f0);
//  rk4_inter <<<1,1>> (U); 
//
//  calculateVolumeRes<<<1,1>>>();
//  calculateFaceRes <<<1,1>>>();
//  addRes2f<<<1,1>>>(f1);
//  rk4_inter <<<1,1>>> (U); 
//
//  calculateVolumeRes<<<1,1>>>();
//  calculateFaceRes <<<1,1>>>();
//  addRes2f<<<1,1>>>(f2);
//  rk4_inter <<<1,1>>> (U); 
//
//  calculateVolumeRes<<<1,1>>>();
//  calculateFaceRes <<<1,1>>>();
//  addRes2f<<<1,1>>>(f3);
//  rk4_inter <<<1,1>>> (U); 
//
//
//  // final rk4 
//  rk4_final <<< 1,1>>> (All, dt, f0,f1,f2,f3); 




  // free memory 
  CUDA_CALL(cudaFree(tempU));  CUDA_CALL(cudaFree(U));
  CUDA_CALL(cudaFree(tempR));  CUDA_CALL(cudaFree(R));
  CUDA_CALL(cudaFree(tempf0)); CUDA_CALL(cudaFree(f0));
  CUDA_CALL(cudaFree(tempf1)); CUDA_CALL(cudaFree(f1));
  CUDA_CALL(cudaFree(tempf2)); CUDA_CALL(cudaFree(f2));
  CUDA_CALL(cudaFree(tempf3)); CUDA_CALL(cudaFree(f3));


  return cudaSuccess;

}
