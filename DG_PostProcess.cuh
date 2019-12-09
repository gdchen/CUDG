#ifndef _DG_PostProcess_H
#define _DG_PostProcess_H

#include <stdlib.h>
#include "DG_All.cuh"
#include "DG_Mesh.cuh"
#include "DG_Basis.cuh"
#include "DG_DataSet.cuh"
#include "DG_Const.cuh"
#include "DG_Math.cuh"

// write out the states at the same lagrange nodes of the solution basis 
int writeStates(const DG_All *All)
{
  int nElem = All->Mesh->nElem;
  int np = All->BasisData->np; 
  DG_DataSet *DataSet = All->DataSet;
  int nGlobal = nElem*np; 
  int i, j;
  double **xyGlobal = (double **)malloc(nGlobal*sizeof(double *));
  for (i=0; i<nGlobal; i++) xyGlobal[i] = (double *)malloc(2*sizeof(double));
  int order = All->BasisData->order;
  getGlobalLagrangeNodes(order, All->Mesh, xyGlobal); 
  FILE *coord, *state;
  coord = fopen("coord.txt", "w");
  state = fopen("state.txt", "w");
  for (i=0; i<nElem; i++){
    for (j=0; j<np; j++){
      fprintf(coord, "%.10f %.10f \n", xyGlobal[i*np+j][0], xyGlobal[i*np+j][1]);
      fprintf(state, "%.10f %.10f %.10f %.10f \n", DataSet->State[i][j*NUM_OF_STATES+0], DataSet->State[i][j*NUM_OF_STATES+1],
                                       DataSet->State[i][j*NUM_OF_STATES+2], DataSet->State[i][j*NUM_OF_STATES+3]);
    }
  }
  fclose(coord);  
  fclose(state);

  for (i=0; i<nGlobal; i++) free(xyGlobal[i]); free(xyGlobal);
  return 0; 
}


// write out the states at the lagrange nodes of order as plotOrder, we 
// can specify higher order than the solution to faithfully represent the 
// high-order solution 
int writeStatesP(const DG_All *All, int plotOrder, char *filename)
{
  int nElem = All->Mesh->nElem;
  //int N = ALl->Mesh->N; // print info 
  int np = All->BasisData->np; 
  int p = All->BasisData->order; // print info 
  //char filename[MAX_CHAR_LENGTH];
  //sprintf(filename, "states_p%d_N%d.txt", p, N);
  int nplot = (plotOrder+1)*(plotOrder+2)/2;
  int i, j, n;
  double **xy = (double **)malloc(nplot*sizeof(double *));
  for (i=0; i<nplot; i++) xy[i] = (double *)malloc(2*sizeof(double));
  getLagrangeNodes(plotOrder, xy);
  // Phi [nplot*np]
  double *Phi = (double *)malloc(nplot*np*sizeof(double));
  for (i=0; i<nplot; i++) DG_TriLagrange(p, xy[i], Phi+i*np);
  // States 
  double **U = (double **)malloc(nElem*sizeof(double *));
  for (i=0; i<nElem; i++) {
    U[i] = (double *)malloc(nplot*NUM_OF_STATES*sizeof(double));
    for (j=0; j<nplot*NUM_OF_STATES; j++) U[i][j] = 0; 
  }

  for (n=0; n<nElem; n++){
    DG_MxM_Set(nplot, np, NUM_OF_STATES, Phi, All->DataSet->State[n], U[n]);
  }

  double **xyGlobal = (double **)malloc(nElem*nplot*sizeof(double *));
  for (n=0; n<nElem*nplot; n++) xyGlobal[n] = (double *)malloc(2*sizeof(double));
  getGlobalLagrangeNodes(plotOrder, All->Mesh, xyGlobal); 
  
  FILE *state;
  state = fopen(filename, "w");
  for (i=0; i<nElem; i++){
    for (j=0; j<nplot; j++){
      fprintf(state, "%.15E %.15E ", xyGlobal[i*nplot+j][0], xyGlobal[i*nplot+j][1]);
      fprintf(state, "%.15E %.15E %.15E %.15E \n", U[i][j*NUM_OF_STATES+0], U[i][j*NUM_OF_STATES+1],
                                       U[i][j*NUM_OF_STATES+2], U[i][j*NUM_OF_STATES+3]);
    }
  }
  //fclose(coord);
  fclose(state);

  for (i=0; i<nplot; i++) free(xy[i]);  free(xy);
  free(Phi);  
  for (i=0; i<nElem; i++)  free(U[i]);  free(U);
  for (i=0; i<nElem*nplot; i++)  free(xyGlobal[i]);  free(xyGlobal);
  return 0; 
}


// measure error with the same quad points used in the solver 
// integrate up to order of 2p+1, p is the solution order
int ErrEst(const DG_All *All, double *err_u, double *err_s, double *err_p)
{
  int nElem = All->Mesh->nElem;
  int nq2 = All->BasisData->nq2;
  int n, i;
  // Global Quad 
  double **xyGlobal = (double **)malloc(nElem*nq2*sizeof(double *));
  for (i=0; i<nElem*nq2; i++) xyGlobal[i] = (double *)malloc(2*sizeof(double));
  getGlobalQuadPoints(xyGlobal, All->Mesh, All->BasisData);
  
  // Quad info 
  double *wq2 = All->BasisData->wq2; 
  double *Uxy = (double *)malloc(nq2*NUM_OF_STATES*sizeof(double));
  for (i=0; i<nq2*NUM_OF_STATES; i++) Uxy[i] = 0; 

  *err_u = 0; *err_s = 0;  *err_p = 0; // initilization
  double f0, f1, f2;
  double rho, rho_exact;
  double rhou, u_exact;
  double rhov, v_exact;
  double rhoe, rhoe_exact;
  double p, p_exact;  
  double sum_erru = 0, sum_errs = 0, sum_errp = 0;
  double r_square = 0; 

/*  FILE *coord, *state;
  coord = fopen("coord.txt", "w");
  state = fopen("state.txt", "w");*/
  for (n=0; n<nElem; n++){
    // Interpolation of the states 
    getIntQuadStates(Uxy, All->DataSet->State[n], All->BasisData);
    // exact solution 
    sum_erru = 0;  sum_errs = 0;  sum_errp = 0;
    for (i=0; i<nq2; i++){

/*      fprintf(coord, "%f %f \n", xyGlobal[n*nq2+i][0], xyGlobal[n*nq2+i][1]);
      fprintf(state, "%f %f %f %f \n", Uxy[i*NUM_OF_STATES+0], Uxy[i*NUM_OF_STATES+1],
                                       Uxy[i*NUM_OF_STATES+2], Uxy[i*NUM_OF_STATES+3]);*/
      f0 = getf0(xyGlobal[n*nq2+i]);
      f1 = getf1(f0);
      f2 = getf2(f0);
      rho_exact = RHO_INF*pow(f1, 1.0/(GAMMA-1));
      u_exact = U_INF - f2*(xyGlobal[n*nq2+i][1]-X_ORIGINAL[1]);
      v_exact = V_INF + f2*(xyGlobal[n*nq2+i][0]-X_ORIGINAL[0]);
      p_exact = P_INF*pow(f1, GAMMA/(GAMMA-1));
      rhoe_exact = p_exact/(GAMMA-1) + 0.5*rho_exact*(u_exact*u_exact+v_exact*v_exact);
      rho = Uxy[i*NUM_OF_STATES+0];
      rhou = Uxy[i*NUM_OF_STATES+1];
      rhov = Uxy[i*NUM_OF_STATES+2];
      rhoe = Uxy[i*NUM_OF_STATES+3];
      p = (GAMMA-1)*(rhoe-0.5*(rhou*rhou+rhov*rhov)/rho); 
      sum_erru += ((rho-rho_exact)*(rho-rho_exact)+(rhou-rho_exact*u_exact)*(rhou-rho_exact*u_exact)
                 + (rhov-rho_exact*v_exact)*(rhov-rho_exact*v_exact) 
                 + (rhoe-rhoe_exact)*(rhoe-rhoe_exact))*wq2[i];
      sum_errs += (log(p/P_INF)-GAMMA*log(rho/RHO_INF))*(log(p/P_INF)-GAMMA*log(rho/RHO_INF))*wq2[i];
      //sum_errs += (log(p_exact/P_INF)-GAMMA*log(rho_exact/RHO_INF))*(log(p_exact/P_INF)-GAMMA*log(rho_exact/RHO_INF))*wq2[i];
      r_square = xyGlobal[n*nq2+i][0]*xyGlobal[n*nq2+i][0] + xyGlobal[n*nq2+i][1]*xyGlobal[n*nq2+i][1];
      sum_errp += r_square*r_square*exp(-2*r_square)*(p-p_exact)*wq2[i];
    }
    *err_u += sum_erru*All->Mesh->detJ[n];
    *err_s += sum_errs*All->Mesh->detJ[n];
    *err_p += sum_errp*All->Mesh->detJ[n];

  }
  for (i=0; i<nElem*nq2; i++) free(xyGlobal[i]);  free(xyGlobal);
  free(Uxy);
 /* fclose(coord);
  fclose(state);*/

  return 0;


} 



// measure error with the higher order quad points,  
// integrate up to order of 2*errestOrder+1, errestOrder is 
// the specified order to integrate the error 
int ErrEstP(const DG_All *All, int errestOrder, double *err_u, double *err_s, double *err_p)
{
  int nElem = All->Mesh->nElem;
  int order = All->BasisData->order;
  //int nq2 = All->BasisData->nq2;
  int nq2; 
  double *xyq; 
  double *wq2; 
  DG_QuadTriangle(2*errestOrder+1, &(nq2), &(xyq), &(wq2));
  int n, i, j;
  // Global Quad 
  double **xyGlobal = (double **)malloc(nElem*nq2*sizeof(double *));
  for (i=0; i<nElem*nq2; i++) xyGlobal[i] = (double *)malloc(2*sizeof(double));
  double **coord = All->Mesh->coord;
  int **E2N = All->Mesh->E2N;
  double **Jac = All->Mesh->Jac;
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
  int np = All->BasisData->np; 
  double *Phi = (double *) malloc(nq2*np*sizeof(double));
  for (i=0; i<nq2; i++){
    xy = xyq+2*i;
    DG_TriLagrange(order, xy, Phi+i*np);
  }
  // Quad info 
  //double *wq2 = All->BasisData->wq2; 
  double *Uxy = (double *)malloc(nq2*NUM_OF_STATES*sizeof(double));
  for (i=0; i<nq2*NUM_OF_STATES; i++) Uxy[i] = 0; 

  *err_u = 0; *err_s = 0;  *err_p = 0; // initilization
  double f0, f1, f2;
  double rho, rho_exact;
  double rhou, u_exact;
  double rhov, v_exact;
  double rhoe, rhoe_exact;
  double p, p_exact;  
  double sum_erru = 0, sum_errs = 0, sum_errp = 0;
  double r_square = 0; 

/*  FILE *coord, *state;
  coord = fopen("coord.txt", "w");
  state = fopen("state.txt", "w");*/
  for (n=0; n<nElem; n++){
    // Interpolation of the states 
    DG_MxM_Set(nq2, np, NUM_OF_STATES, Phi, All->DataSet->State[n], Uxy);
    //getIntQuadStates(Uxy, All->DataSet->State[n], All->BasisData);
    // exact solution 
    sum_erru = 0;  sum_errs = 0;  sum_errp = 0;
    for (i=0; i<nq2; i++){

/*      fprintf(coord, "%f %f \n", xyGlobal[n*nq2+i][0], xyGlobal[n*nq2+i][1]);
      fprintf(state, "%f %f %f %f \n", Uxy[i*NUM_OF_STATES+0], Uxy[i*NUM_OF_STATES+1],
                                       Uxy[i*NUM_OF_STATES+2], Uxy[i*NUM_OF_STATES+3]);*/
      f0 = getf0(xyGlobal[n*nq2+i]);
      f1 = getf1(f0);
      f2 = getf2(f0);
      rho_exact = RHO_INF*pow(f1, 1.0/(GAMMA-1));
      u_exact = U_INF - f2*(xyGlobal[n*nq2+i][1]-X_ORIGINAL[1]);
      v_exact = V_INF + f2*(xyGlobal[n*nq2+i][0]-X_ORIGINAL[0]);
      p_exact = P_INF*pow(f1, GAMMA/(GAMMA-1));
      rhoe_exact = p_exact/(GAMMA-1) + 0.5*rho_exact*(u_exact*u_exact+v_exact*v_exact);
      rho = Uxy[i*NUM_OF_STATES+0];
      rhou = Uxy[i*NUM_OF_STATES+1];
      rhov = Uxy[i*NUM_OF_STATES+2];
      rhoe = Uxy[i*NUM_OF_STATES+3];
      p = (GAMMA-1)*(rhoe-0.5*(rhou*rhou+rhov*rhov)/rho); 
      sum_erru += ((rho-rho_exact)*(rho-rho_exact)+(rhou-rho_exact*u_exact)*(rhou-rho_exact*u_exact)
                 + (rhov-rho_exact*v_exact)*(rhov-rho_exact*v_exact) 
                 + (rhoe-rhoe_exact)*(rhoe-rhoe_exact))*wq2[i];
      sum_errs += (log(p/P_INF)-GAMMA*log(rho/RHO_INF))*(log(p/P_INF)-GAMMA*log(rho/RHO_INF))*wq2[i];
      //sum_errs += (log(p_exact/P_INF)-GAMMA*log(rho_exact/RHO_INF))*(log(p_exact/P_INF)-GAMMA*log(rho_exact/RHO_INF))*wq2[i];
      r_square = xyGlobal[n*nq2+i][0]*xyGlobal[n*nq2+i][0] + xyGlobal[n*nq2+i][1]*xyGlobal[n*nq2+i][1];
      sum_errp += r_square*r_square*exp(-2*r_square)*(p-p_exact)*wq2[i];
    }
    *err_u += sum_erru*All->Mesh->detJ[n];
    *err_s += sum_errs*All->Mesh->detJ[n];
    *err_p += sum_errp*All->Mesh->detJ[n];

  }
  for (i=0; i<nElem*nq2; i++) free(xyGlobal[i]);  free(xyGlobal);
  free(Uxy);
  free(xyq);
  free(wq2);
  free(Phi);
 /* fclose(coord);
  fclose(state);*/

  return 0;


} 










#endif
